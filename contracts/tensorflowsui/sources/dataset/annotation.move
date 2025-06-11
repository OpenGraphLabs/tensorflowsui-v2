// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

module tensorflowsui::annotation {
  use std::string::{Self, String};
  use sui::address;

  public enum Annotation has copy, drop, store {
    Label {
      path: String,
      status: AnnotationStatus,
      value: LabelAnnotation,
    },
    BBox {
      path: String,
      status: AnnotationStatus,
      value: BBoxAnnotation,
    },
    Skeleton {
      path: String,
      status: AnnotationStatus,
      value: SkeletonAnnotation,
    },
  }

  public enum AnnotationStatus has copy, drop, store {
    Pending,
    Confirmed,
    Rejected,
  }

  public struct LabelAnnotation has copy, drop, store {
    label: String,
    annotated_by: address,
  }

  public struct BBoxAnnotation has copy, drop, store {
    // coordinates of the bounding box [x, y, w, h]
    x: u64,
    y: u64,
    w: u64,
    h: u64,
    annotated_by: address,
  }

  public struct SkeletonAnnotation has copy, drop, store {
    keypoints: vector<Point>, // keypoints of the skeleton [(x1,y1), (x2,y2), ...]
    edges: vector<Edge>,  // edges connecting keypoints [(start_idx, end_idx), ...]
    annotated_by: address,
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

  public fun new_annotation_path(annotation: Annotation, ctx: &mut TxContext): AnnotationPath {
    // annotation path key format `address::value`
    let mut path = address::to_string(ctx.sender());
    path.append(string::utf8(b"::"));
    match (annotation) {
      Annotation::Label { path: _, status: _, value } => {
        path.append(value.label);
      },
      Annotation::BBox { path: _, status: _, value } => {
        // format: x,y,w,h
        path.append(value.x.to_string());
        path.append_utf8(b",");
        path.append(value.y.to_string());
        path.append_utf8(b",");
        path.append(value.w.to_string());
        path.append_utf8(b",");
        path.append(value.h.to_string());
      },
      Annotation::Skeleton { path: _, status: _, value } => {
        // format: keypoints,edges
        // TODO(jarry): format keypoints and edges
        path.append(string::utf8(b"keypoints"));
        path.append_utf8(b",");
        path.append(string::utf8(b"edges"));
      }
    };

    AnnotationPath { path }
  }

  public fun new_label_annotation(path: String, label: String, annotated_by: address): Annotation {
    Annotation::Label { path, value: LabelAnnotation { label, annotated_by }, status: AnnotationStatus::Pending }
  }

  public fun new_bbox_annotation(path: String, x: u64, y: u64, w: u64, h: u64, annotated_by: address): Annotation {
    Annotation::BBox { path, value: BBoxAnnotation { x, y, w, h, annotated_by }, status: AnnotationStatus::Pending }
  }

  public fun new_skeleton_annotation(path: String, keypoints: vector<Point>, edges: vector<Edge>, annotated_by: address): Annotation {
    // TODD(jarry): logic for initializing keypoints and edges from primitive inputs
    Annotation::Skeleton { path, value: SkeletonAnnotation { keypoints, edges, annotated_by }, status: AnnotationStatus::Pending }
  }

  public fun approve_annotation(annotation: &mut Annotation) {
    match (annotation) {  
      Annotation::Label { path: _, status, value: _ } => {
        *status = AnnotationStatus::Confirmed;
      },
      Annotation::BBox { path: _, status, value: _ } => {
        *status = AnnotationStatus::Confirmed; 
      },
      Annotation::Skeleton { path: _, status, value: _ } => {
        *status = AnnotationStatus::Confirmed;
      },
    }
  }

  public fun reject_annotation(annotation: &mut Annotation) {
    match (annotation) {
      Annotation::Label { path: _, status, value: _ } => {
        *status = AnnotationStatus::Rejected;
      },
      Annotation::BBox { path: _, status, value: _ } => {
        *status = AnnotationStatus::Rejected;
      },
      Annotation::Skeleton { path: _, status, value: _ } => {
        *status = AnnotationStatus::Rejected;
      }
    }
  }
}
