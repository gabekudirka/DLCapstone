import requests
import shutil
import json
from pyproj import CRS, Transformer
import math
import numpy as np
import csv
import pandas as pd
from scipy import spatial

class RoadSegment():
    def __init__(
            self, 
            route_id,
            geometry):
        self.route_id = route_id
        self.num_pts = len(geometry)
        
        self.pts = [(coords[1], coords[0]) for coords in geometry]

        #Project the points from road data onto the same projection as aerial imagery
        crs_4326 = CRS('epsg:4326')
        crs_proj = CRS('epsg:26985')
        transformer = Transformer.from_crs(crs_4326, crs_proj)
        pts_proj = transformer.itransform(self.pts)
        self.pts_proj = [pt for pt in pts_proj]
        
        #Calculate the distance between each section in the segment
        self.sub_distances = []
        for i, coords in enumerate(self.pts_proj):
            if i == len(self.pts_proj) - 1:
                break
            x1, y1 = coords
            x2, y2 = self.pts_proj[i+1]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            self.sub_distances.append(distance)
        self.total_distance = sum(self.sub_distances)

#find a point some distance between p1 and p2
def interp_pts(p1, p2, dist):
    x1, y1 = p1
    x2, y2 = p2
    total_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    x3 = x1 + (dist / total_dist) * (x2 - x1)
    y3 = y1 + (dist / total_dist) * (y2 - y1)

    return x3, y3

def collect_img_origins(segment, img_dim, overlap):
    img_origins = [segment.pts_proj[0], segment.pts_proj[-1]]
    #automatically want images of both ends of segment subtract image dim to account for this
    #divide by the dimension of an image minus overlap to determine number of images to take of segment
    num_imgs = math.ceil((segment.total_distance - img_dim + (overlap * 2)) / (img_dim - (overlap * 2)))
    if num_imgs == 0:
        return img_origins

    #Since we're rounding up, adjust the increment to evenly space out the images
    increment = segment.total_distance / (num_imgs + 1)
    #Find the distance from the start that each image should be taken at
    img_distances = [(i+1) * increment for i in range(0, num_imgs)]    
    #Find the section that each image should be centered around
    sections = []
    section_idx = 0
    section_distance = segment.sub_distances[section_idx]
    for distance in img_distances:
        while distance > section_distance:
            section_idx += 1
            section_distance += segment.sub_distances[section_idx]
        sections.append((distance, section_idx))
    dist_accumulator = 0
    accumulated_dists = [0]
    for dist in segment.sub_distances:
        dist_accumulator += dist
        accumulated_dists.append(dist_accumulator)
    #Find the center point that each image should be taken around
    for distance, section_idx in sections:
        p1 = segment.pts_proj[section_idx]
        p2 = segment.pts_proj[section_idx + 1]
        dist = distance - accumulated_dists[section_idx]
        img_pt = interp_pts(p1, p2, dist)
        img_origins.append(img_pt)

    return img_origins

def find_clusters_kd(img_coords, min_dist):
    kd_tree = spatial.KDTree(img_coords)
    clusters_raw = kd_tree.query_ball_point(img_coords, min_dist)
    #Remove duplicates
    clusters_set = {tuple(cluster) for cluster in clusters_raw}
    return clusters_set

def merge_clusters(clusters, img_coords):
    new_coords = []
    clustered_pts = set()
    for cluster in clusters:
        mean_x = 0
        mean_y = 0
        for pt_idx in cluster:
            x, y = img_coords[pt_idx]
            mean_x += x
            mean_y += y
            clustered_pts.add(pt_idx)
        mean_x /= len(cluster)
        mean_y /= len(cluster)        
        new_coords.append((mean_x, mean_y))
    
    coords_idxs = set(np.arange(0, len(img_coords), 1))
    non_clusterd_pts = coords_idxs.difference(clustered_pts)
    for pt_idx in non_clusterd_pts:
        new_coords.append(img_coords[pt_idx])
    return np.asarray(new_coords)

def project_to_latlng(pt):
    crs_4326 = CRS('epsg:4326')
    crs_proj = CRS('epsg:26985')
    transformer = Transformer.from_crs(crs_proj, crs_4326)
    pt_proj = transformer.transform(pt[0], pt[1])
    
    return pt_proj

def convert_to_bbox(img_coord, dim):
    r = dim / 2
    xmin = img_coord[0] - r
    xmax = img_coord[0] + r
    ymin = img_coord[1] - r
    ymax = img_coord[1] + r
    
    return (xmin, ymin, xmax, ymax)

if __name__ == '__main__':
    with open('./data/roadways/Roadway_Block.geojson') as f:
        dc_roadway_data = json.load(f)

    #Process the road network data
    dc_road_segments = []
    for segment in dc_roadway_data['features']:
        if segment['geometry']['type'] == 'MultiLineString':
            for LineString in segment['geometry']['coordinates']:
                segment_obj = RoadSegment(
                    segment['properties']['ROUTEID'],
                    LineString
                )
                dc_road_segments.append(segment_obj)
        else:
            segment_obj = RoadSegment(
                        segment['properties']['ROUTEID'],
                        segment['geometry']['coordinates']
            )
            dc_road_segments.append(segment_obj)

    BBOX_DIM = 60
    OVERLAP = 6
    MAX_DIST = 30

    #Collect evenly spaced coordinates along the road network for images
    all_img_coords = []
    for segment in dc_road_segments:
        seg_img_coords = collect_img_origins(segment, BBOX_DIM, OVERLAP)
        all_img_coords.extend(seg_img_coords)
    all_img_coords = np.asarray(all_img_coords)

    #Merge cooredinates with too much overlap
    coord_clusters = find_clusters_kd(all_img_coords, MAX_DIST)
    merged_coords = merge_clusters(coord_clusters, all_img_coords)
    coord_clusters = find_clusters_kd(merged_coords, MAX_DIST)
    merged_coords = merge_clusters(coord_clusters, merged_coords)

    #Filter coordinates to one defined region
    latlng_coords = [project_to_latlng(coord) for coord in merged_coords]
    image_bboxes = [(convert_to_bbox(coord, BBOX_DIM), latlng) for coord, latlng in zip(merged_coords, latlng_coords)]
    region_bbox = (399697, 135518, 401430, 136935)
    bboxes_in_region = [coord for coord in image_bboxes if coord[0][0] > region_bbox[0] and coord[0][1] > region_bbox[1]
                        and coord[0][2] < region_bbox[2] and coord[0][3] < region_bbox[3]]
    latlng_in_region = [coord[1] for coord in bboxes_in_region]
    bboxes_in_region = [coord[0] for coord in bboxes_in_region]

    df_region = pd.DataFrame(bboxes_in_region, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    image_filenames = ['image_' + str(i) + '.png' for i in range(len(bboxes_in_region))]
    df_region['filename'] = image_filenames
    df_region['lat'] = [coord[0] for coord in latlng_in_region]
    df_region['lng'] = [coord[1] for coord in latlng_in_region]

    df_region.to_csv('region_image_coords.csv', index=False)

    