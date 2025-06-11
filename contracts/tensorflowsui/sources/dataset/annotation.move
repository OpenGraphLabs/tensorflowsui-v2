// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

module tensorflowsui::annotation {
  use std::string::{String};
  use sui::address;

  public struct AnnotationStatus has copy, drop, store {
    is_confirmed: bool,
    confirmed_at: Option<u64>,  // timestamp when confirmed
    confirmed_by: Option<address>,  // address of the confirmer
  }

  public struct LabelAnnotation has copy, drop, store {
    label: String,
    annotated_by: address,
    status: AnnotationStatus,
  }

  public struct BBoxAnnotation has copy, drop, store {
    // coordinates of the bounding box [x, y, w, h]
    x: u64,
    y: u64,
    w: u64,
    h: u64,
    annotated_by: address,
    status: AnnotationStatus,
  }

  public struct SkeletonAnnotation has copy, drop, store {
    keypoints: vector<Point>, // keypoints of the skeleton [(x1,y1), (x2,y2), ...]
    edges: vector<Edge>,  // edges connecting keypoints [(start_idx, end_idx), ...]
    annotated_by: address,
    status: AnnotationStatus,
  }

  public struct Point has copy, drop, store {
    x: u64,
    y: u64,
  }

  public struct Edge has copy, drop, store {
    start_idx: u64,
    end_idx: u64,
  }

  public struct AnnotationPath has copy, drop, store {
    path: String,
  }

  public fun new_annotation_path(value: String, ctx: &mut TxContext): AnnotationPath {
    // annotation path key format `address::value`
    let mut path = address::to_string(ctx.sender());
    path.append_utf8(b"::");
    path.append(value);
    
    AnnotationPath { path }
  }

  public struct Annotation has copy, drop, store {
    path: String,
    // label of the annotation
    label: LabelAnnotation,
    // bounding box of the annotation
    bbox: BBoxAnnotation,
    // skeleton of the annotation
    skeleton: SkeletonAnnotation,
  } 

  public fun new_annotation(
    path: String,
    label: LabelAnnotation,
    bbox: BBoxAnnotation,
    skeleton: SkeletonAnnotation,
  ): Annotation {
    Annotation { path, label, bbox, skeleton }
  }
}
