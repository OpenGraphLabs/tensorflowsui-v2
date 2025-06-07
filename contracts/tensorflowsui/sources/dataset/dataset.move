// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

module tensorflowsui::dataset {
  use std::string::{Self, String};
  use sui::display::{Self, Display};
  use sui::package::{Self, Publisher};
  use sui::vec_map::{Self, VecMap};
  use tensorflowsui::metadata;
  use sui::event;
  use sui::dynamic_field;

  const OPENGRAPH_LICENSE: vector<u8> = b"OpenGraph License";

  /// Error codes
  const EStartAndEndRangeAreNone: u64 = 0;
  const ERangeStartGreaterThanRangeEnd: u64 = 1;
  const EInvalidBatchSize: u64 = 2;

  /// Event emitted when a new dataset is created.
  public struct DatasetCreated has copy, drop {
    dataset_id: ID,
  }

  /// Event emitted when a dataset is burned.
  public struct DatasetBurnedEvent has copy, drop {
    dataset_id: ID,
  }

  /// Emits a DatasetCreated event.
  public fun emit_dataset_created(dataset_id: ID) {
      event::emit(DatasetCreated { dataset_id });
  }

  /// Emits a DatasetBurned event.
  public fun emit_dataset_burned(dataset_id: ID) {
    event::emit(DatasetBurnedEvent {
        dataset_id,
    });
}

  /// The dataset published on Sui.
  public struct Dataset has key, store {
      id: UID,
      name: String,
      description: Option<String>,
      // tags of the dataset
      tags: Option<vector<String>>,
      // type of the data in the dataset (eg. parquet, csv, json, png, jpg, etc.)
      data_type: String,
      // size of the data in the dataset
      data_size: u64,
      // creator of the dataset
      creator: Option<String>,
      // license of the dataset
      license: String,
  }

  /// An annotation type for a data in a dataset.
  public struct LabelAnnotation has copy, drop, store {
    // label of the annotation
    label: String,
    // address of the user who created this annotation
    annotated_by: address,
  }

  /// A bounding box annotation for a data in a dataset.
  public struct BBoxAnnotation has copy, drop, store {
    // coordinates of the bounding box [x1, y1, x2, y2]
    x1: u64,
    y1: u64,
    x2: u64,
    y2: u64,
    // address of the user who created this annotation
    annotated_by: address,
  }

  /// A skeleton annotation for a data in a dataset.
  public struct SkeletonAnnotation has copy, drop, store {
    // keypoints of the skeleton [(x1,y1), (x2,y2), ...]
    keypoints: vector<Point>,
    // edges connecting keypoints [(start_idx, end_idx), ...]
    edges: vector<Edge>,
    // address of the user who created this annotation
    annotated_by: address,
  }

  /// A point struct for skeleton keypoints
  public struct Point has copy, drop, store {
    x: u64,
    y: u64,
  }

  /// An edge struct for skeleton connections
  public struct Edge has copy, drop, store {
    start_idx: u64,
    end_idx: u64,
  }

  /// A data in a dataset.
  public struct Data has drop, store {
      path: String,

      // The walrus blob id containing the bytes for this resource.
      blob_id: String,

      // Contains the hash of the contents of the blob
      // to verify its integrity.
      blob_hash: String,

      // The type of the data in the dataset
      data_type: String,

      // Defines the byte range of the resource contents
      // in the case where multiple resources are stored
      // in the same blob. This way, each resource will
      // be parsed using its' byte range in the blob.
      range: Option<Range>,

      // Pending annotation statistics
      pending_label_stats: VecMap<String, vector<address>>,
      pending_bbox_stats: VecMap<vector<u64>, vector<address>>,
      pending_skeleton_stats: VecMap<vector<u64>, vector<address>>,

      // Confirmed annotations
      confirmed_label_annotations: vector<LabelAnnotation>,
      confirmed_bbox_annotations: vector<BBoxAnnotation>,
      confirmed_skeleton_annotations: vector<SkeletonAnnotation>,
  }

  /// An annotation for a data in a dataset.
  public struct Annotation has copy, drop, store {
    // label of the annotation
    label: String,
  }

  public struct Range has drop, store {
      start: Option<u64>, // inclusive lower bound
      end: Option<u64>, // exclusive upper bound
  }

  /// Representation of the data path.
  /// Ensures there are no namespace collisions in the dynamic fields.
  public struct DataPath has copy, drop, store {
      path: String,
  }

  fun new_data_path(path: String): DataPath {
    DataPath { path }
  }

  /// One-Time-Witness for the module.
  public struct DATASET has drop {}

  fun init(otw: DATASET, ctx: &mut TxContext) {
    let publisher = package::claim(otw, ctx);
    let d = init_dataset_display(&publisher, ctx);
    transfer::public_transfer(d, ctx.sender());
    transfer::public_transfer(publisher, ctx.sender());
  }

  /// Creates a new dataset.
  public fun new_dataset(name: String, metadata: metadata::Metadata, ctx: &mut TxContext): Dataset {
    let license: String = option::get_with_default(
      &metadata::license(&metadata), 
      string::utf8(OPENGRAPH_LICENSE),
    );

    let dataset = Dataset {
        id: object::new(ctx),
        name,
        description: metadata::description(&metadata),
        data_type: metadata::data_type(&metadata),
        data_size: metadata::data_size(&metadata),
        creator: metadata::creator(&metadata),
        license: license,
        tags: metadata::tags(&metadata),
    };
    emit_dataset_created(
        object::id(&dataset),
    );
    
    dataset
  }

  /// Optionally creates a new Range object.
  public fun new_range_option(range_start: Option<u64>, range_end: Option<u64>): Option<Range> {
      if (range_start.is_none() && range_end.is_none()) {
          return option::none<Range>()
      };
      option::some(new_range(range_start, range_end))
  }

  /// Creates a new Range object.
  ///
  /// Aborts if both range_start and range_end are none.
  /// Aborts if the range_start is greater than the range_end.
  public fun new_range(range_start: Option<u64>, range_end: Option<u64>): Range {
      let start_is_defined = range_start.is_some();
      let end_is_defined = range_end.is_some();

      // At least one of the range bounds should be defined.
      assert!(start_is_defined || end_is_defined, EStartAndEndRangeAreNone);

      // If both range bounds are defined, the upper bound should be greater than the lower.
      if (start_is_defined && end_is_defined) {
          let start = option::borrow(&range_start);
          let end = option::borrow(&range_end);
          assert!(*end > *start, ERangeStartGreaterThanRangeEnd);
      };

      Range {
          start: range_start,
          end: range_end,
      }
  }

  /// Updates the name of a dataset.
  public fun update_name(dataset: &mut Dataset, new_name: String) {
    dataset.name = new_name
  }

  /// Update the site metadata.
  public fun update_metadata(dataset: &mut Dataset, metadata: metadata::Metadata) {
    dataset.description = metadata::description(&metadata);
    dataset.data_type = metadata::data_type(&metadata);
    dataset.data_size = metadata::data_size(&metadata);
    dataset.creator = metadata::creator(&metadata);
  }

  /// Creates a new Data object.
  public fun new_data(
      path: String,
      blob_id: String,
      blob_hash: String,
      data_type: String,
      range: Option<Range>,
  ): Data {
      Data {
          path,
          blob_id,
          blob_hash,
          data_type,
          range,
          pending_label_stats: vec_map::empty<String, vector<address>>(),
          pending_bbox_stats: vec_map::empty<vector<u64>, vector<address>>(),
          pending_skeleton_stats: vec_map::empty<vector<u64>, vector<address>>(),
          confirmed_label_annotations: vector[],
          confirmed_bbox_annotations: vector[],
          confirmed_skeleton_annotations: vector[],
      }
  }

  /// Adds a data to an existing dataset.
  public fun add_data(dataset: &mut Dataset, data: Data) {
    let path_obj = new_data_path(data.path);
    dynamic_field::add(&mut dataset.id, path_obj, data);
  }

  /// Removes a data from a dataset.
  ///
  /// Aborts if the data does not exist.
  public fun remove_data(dataset: &mut Dataset, path: String): Data {
    let path_obj = new_data_path(path);
    dynamic_field::remove(&mut dataset.id, path_obj)
  }

  /// Removes a data from a dataset if it exists.
  public fun remove_data_if_exists(dataset: &mut Dataset, path: String): Option<Data> {
    let path_obj = new_data_path(path);
    dynamic_field::remove_if_exists(&mut dataset.id, path_obj)
  }

  /// Changes the path of a data on a dataset.
  public fun move_data(dataset: &mut Dataset, old_path: String, new_path: String) {
    let mut data = remove_data(dataset, old_path);
    data.path = new_path;
    add_data(dataset, data);
  }

  /// Adds a pending label annotation (user-submitted, not yet validated)
  public fun add_pending_label_annotation(dataset: &mut Dataset, path: String, label: String, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let sender = tx_context::sender(ctx);
    
    // Increment the count for this label
    if (vec_map::contains(&data.pending_label_stats, &label)) {
      let addresses = vec_map::get_mut(&mut data.pending_label_stats, &label);
      vector::push_back(addresses, sender);
    } else {
      vec_map::insert(&mut data.pending_label_stats, label, vector[sender]);
    }
  }

  /// Adds a pending bounding box annotation (user-submitted, not yet validated)
  public fun add_pending_bbox_annotation(dataset: &mut Dataset, path: String, x1: u64, y1: u64, x2: u64, y2: u64, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let sender = tx_context::sender(ctx);
    let coords = vector[x1, y1, x2, y2];
    
    // Increment the count for this bbox
    if (vec_map::contains(&data.pending_bbox_stats, &coords)) {
      let addresses = vec_map::get_mut(&mut data.pending_bbox_stats, &coords);
      vector::push_back(addresses, sender);
    } else {
      vec_map::insert(&mut data.pending_bbox_stats, coords, vector[sender]);
    }
  }

  /// Adds a pending skeleton annotation (user-submitted, not yet validated)
  public fun add_pending_skeleton_annotation(
    dataset: &mut Dataset, 
    path: String, 
    keypoints: vector<Point>,
    edges: vector<Edge>,
    ctx: &mut TxContext
  ) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let sender = tx_context::sender(ctx);
    
    // Convert keypoints and edges to a flat vector of u64s for storage
    let mut flat_data = vector::empty<u64>();
    let mut i = 0;
    while (i < vector::length(&keypoints)) {
      let point = vector::borrow(&keypoints, i);
      vector::push_back(&mut flat_data, point.x);
      vector::push_back(&mut flat_data, point.y);
      i = i + 1;
    };
    
    i = 0;
    while (i < vector::length(&edges)) {
      let edge = vector::borrow(&edges, i);
      vector::push_back(&mut flat_data, edge.start_idx);
      vector::push_back(&mut flat_data, edge.end_idx);
      i = i + 1;
    };
    
    // Increment the count for this skeleton
    if (vec_map::contains(&data.pending_skeleton_stats, &flat_data)) {
      let addresses = vec_map::get_mut(&mut data.pending_skeleton_stats, &flat_data);
      vector::push_back(addresses, sender);
    } else {
      vec_map::insert(&mut data.pending_skeleton_stats, flat_data, vector[sender]);
    }
  }

  /// Adds pending label annotations in batch using two vectors (paths and labels must have same length)
  public fun batch_add_pending_label_annotations(
    dataset: &mut Dataset, 
    paths: vector<String>, 
    labels: vector<String>, 
    ctx: &mut TxContext
  ) {
    // Ensure both vectors have the same length
    assert!(vector::length(&paths) == vector::length(&labels), EInvalidBatchSize);
    
    let mut i = 0;
    let len = vector::length(&paths);
    
    while (i < len) {
      let path = *vector::borrow(&paths, i);
      let label = *vector::borrow(&labels, i);
      add_pending_label_annotation(dataset, path, label, ctx);
      i = i + 1;
    }
  }

  /// Adds pending bounding box annotations in batch
  public fun batch_add_pending_bbox_annotations(
    dataset: &mut Dataset, 
    paths: vector<String>, 
    coords: vector<vector<u64>>, 
    ctx: &mut TxContext
  ) {
    // Ensure both vectors have the same length
    assert!(vector::length(&paths) == vector::length(&coords), EInvalidBatchSize);
    
    let mut i = 0;
    let len = vector::length(&paths);
    
    while (i < len) {
      let path = *vector::borrow(&paths, i);
      let coord = vector::borrow(&coords, i);
      add_pending_bbox_annotation(
        dataset, 
        path, 
        *vector::borrow(coord, 0), // x1
        *vector::borrow(coord, 1), // y1
        *vector::borrow(coord, 2), // x2
        *vector::borrow(coord, 3), // y2
        ctx
      );
      i = i + 1;
    }
  }

  /// Adds pending skeleton annotations in batch
  public fun batch_add_pending_skeleton_annotations(
    dataset: &mut Dataset, 
    paths: vector<String>, 
    keypoints: vector<vector<Point>>,
    edges: vector<vector<Edge>>,
    ctx: &mut TxContext
  ) {
    // Ensure all vectors have the same length
    assert!(
      vector::length(&paths) == vector::length(&keypoints) && 
      vector::length(&paths) == vector::length(&edges), 
      EInvalidBatchSize
    );
    
    let mut i = 0;
    let len = vector::length(&paths);
    
    while (i < len) {
      let path = *vector::borrow(&paths, i);
      add_pending_skeleton_annotation(
        dataset,
        path,
        *vector::borrow(&keypoints, i),
        *vector::borrow(&edges, i),
        ctx
      );
      i = i + 1;
    }
  }

  /// Gets the count of a specific pending label annotation
  public fun get_pending_label_annotation_count(dataset: &Dataset, path: String, label: String): u64 {
    let data = dynamic_field::borrow<DataPath, Data>(&dataset.id, new_data_path(path));
    if (vec_map::contains(&data.pending_label_stats, &label)) {
      vector::length(vec_map::get(&data.pending_label_stats, &label))
    } else {
      0
    }
  }

  /// Gets the count of a specific pending bounding box annotation
  public fun get_pending_bbox_annotation_count(dataset: &Dataset, path: String, x1: u64, y1: u64, x2: u64, y2: u64): u64 {
    let data = dynamic_field::borrow<DataPath, Data>(&dataset.id, new_data_path(path));
    let coords = vector[x1, y1, x2, y2];
    if (vec_map::contains(&data.pending_bbox_stats, &coords)) {
      vector::length(vec_map::get(&data.pending_bbox_stats, &coords))
    } else {
      0
    }
  }

  /// Gets the count of a specific pending skeleton annotation
  public fun get_pending_skeleton_annotation_count(dataset: &Dataset, path: String, keypoints: vector<Point>, edges: vector<Edge>): u64 {
    let data = dynamic_field::borrow<DataPath, Data>(&dataset.id, new_data_path(path));
    
    // Convert keypoints and edges to a flat vector of u64s for lookup
    let mut flat_data = vector::empty<u64>();
    let mut i = 0;
    while (i < vector::length(&keypoints)) {
      let point = vector::borrow(&keypoints, i);
      vector::push_back(&mut flat_data, point.x);
      vector::push_back(&mut flat_data, point.y);
      i = i + 1;
    };
    
    i = 0;
    while (i < vector::length(&edges)) {
      let edge = vector::borrow(&edges, i);
      vector::push_back(&mut flat_data, edge.start_idx);
      vector::push_back(&mut flat_data, edge.end_idx);
      i = i + 1;
    };
    
    if (vec_map::contains(&data.pending_skeleton_stats, &flat_data)) {
      vector::length(vec_map::get(&data.pending_skeleton_stats, &flat_data))
    } else {
      0
    }
  }

  /// Gets all pending label annotations and their annotators
  public fun get_all_pending_label_annotations(dataset: &Dataset, path: String): (vector<String>, vector<vector<address>>) {
    let data = dynamic_field::borrow<DataPath, Data>(&dataset.id, new_data_path(path));
    let labels = vec_map::keys(&data.pending_label_stats);
    let mut annotators = vector::empty<vector<address>>();
    let mut i = 0;
    while (i < vector::length(&labels)) {
      let label = vector::borrow(&labels, i);
      vector::push_back(&mut annotators, *vec_map::get(&data.pending_label_stats, label));
      i = i + 1;
    };
    (labels, annotators)
  }

  /// Gets all pending bounding box annotations and their annotators
  public fun get_all_pending_bbox_annotations(dataset: &Dataset, path: String): (vector<vector<u64>>, vector<vector<address>>) {
    let data = dynamic_field::borrow<DataPath, Data>(&dataset.id, new_data_path(path));
    let coords = vec_map::keys(&data.pending_bbox_stats);
    let mut annotators = vector::empty<vector<address>>();
    let mut i = 0;
    while (i < vector::length(&coords)) {
      let coord = vector::borrow(&coords, i);
      vector::push_back(&mut annotators, *vec_map::get(&data.pending_bbox_stats, coord));
      i = i + 1;
    };
    (coords, annotators)
  }

  /// Gets all pending skeleton annotations and their annotators
  public fun get_all_pending_skeleton_annotations(dataset: &Dataset, path: String): (vector<vector<u64>>, vector<vector<address>>) {
    let data = dynamic_field::borrow<DataPath, Data>(&dataset.id, new_data_path(path));
    let flat_data = vec_map::keys(&data.pending_skeleton_stats);
    let mut annotators = vector::empty<vector<address>>();
    let mut i = 0;
    while (i < vector::length(&flat_data)) {
      let data_point = vector::borrow(&flat_data, i);
      vector::push_back(&mut annotators, *vec_map::get(&data.pending_skeleton_stats, data_point));
      i = i + 1;
    };
    (flat_data, annotators)
  }

  /// Validates pending label annotations and promotes them to confirmed label annotations
  public fun validate_label_annotations(dataset: &mut Dataset, path: String, labels_to_confirm: vector<String>, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    while (i < vector::length(&labels_to_confirm)) {
      let label = vector::borrow(&labels_to_confirm, i);
      if (vec_map::contains(&data.pending_label_stats, label)) {
        let annotators = vec_map::get(&data.pending_label_stats, label);
        let mut j = 0;
        while (j < vector::length(annotators)) {
          data.confirmed_label_annotations.push_back(LabelAnnotation { 
            label: *label,
            annotated_by: *vector::borrow(annotators, j)
          });
          j = j + 1;
        }
      };
      i = i + 1;
    }
  }

  /// Validates pending bounding box annotations and promotes them to confirmed annotations
  public fun validate_bbox_annotations(dataset: &mut Dataset, path: String, coords_to_confirm: vector<vector<u64>>) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    while (i < vector::length(&coords_to_confirm)) {
      let coords = vector::borrow(&coords_to_confirm, i);
      if (vec_map::contains(&data.pending_bbox_stats, coords)) {
        let annotators = vec_map::get(&data.pending_bbox_stats, coords);
        let mut j = 0;
        while (j < vector::length(annotators)) {
          data.confirmed_bbox_annotations.push_back(BBoxAnnotation { 
            x1: *vector::borrow(coords, 0),
            y1: *vector::borrow(coords, 1),
            x2: *vector::borrow(coords, 2),
            y2: *vector::borrow(coords, 3),
            annotated_by: *vector::borrow(annotators, j)
          });
          j = j + 1;
        }
      };
      i = i + 1;
    }
  }

  /// Validates pending skeleton annotations and promotes them to confirmed annotations
  public fun validate_skeleton_annotations(
    dataset: &mut Dataset, 
    path: String, 
    flat_data_to_confirm: vector<vector<u64>>
  ) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    while (i < vector::length(&flat_data_to_confirm)) {
      let flat_data = vector::borrow(&flat_data_to_confirm, i);
      if (vec_map::contains(&data.pending_skeleton_stats, flat_data)) {
        let annotators = vec_map::get(&data.pending_skeleton_stats, flat_data);
        let mut j = 0;
        while (j < vector::length(annotators)) {
          // Reconstruct keypoints and edges from flat data
          let num_points = vector::length(flat_data) / 4; // Each point is 2 values, each edge is 2 values
          let mut keypoints = vector::empty<Point>();
          let mut edges = vector::empty<Edge>();
          let mut k = 0;
          while (k < num_points) {
            vector::push_back(&mut keypoints, Point {
              x: *vector::borrow(flat_data, k * 2),
              y: *vector::borrow(flat_data, k * 2 + 1)
            });
            k = k + 1;
          };
          k = num_points * 2;
          while (k < vector::length(flat_data)) {
            vector::push_back(&mut edges, Edge {
              start_idx: *vector::borrow(flat_data, k),
              end_idx: *vector::borrow(flat_data, k + 1)
            });
            k = k + 2;
          };
          
          data.confirmed_skeleton_annotations.push_back(SkeletonAnnotation { 
            keypoints,
            edges,
            annotated_by: *vector::borrow(annotators, j)
          });
          j = j + 1;
        }
      };
      i = i + 1;
    }
  }

  /// Deletes a dataset object.
  ///
  /// NB: This function does **NOT** delete the dynamic fields! Make sure to call this function
  /// after deleting manually all the dynamic fields attached to the dataset object. If you don't
  /// delete the dynamic fields, they will become unaccessible and you will not be able to delete
  /// them in the future.
  public fun burn(dataset: Dataset) {
    emit_dataset_burned(object::id(&dataset));
    let Dataset {
        id,
        ..,
    } = dataset;
    id.delete();
  }

  /// Define a Display for the Dataset objects.
  fun init_dataset_display(publisher: &Publisher, ctx: &mut TxContext): Display<Dataset> {
      let keys = vector[
          b"name".to_string(),
          b"description".to_string(),
          b"tags".to_string(),
          b"data_type".to_string(),
          b"data_size".to_string(),
          b"creator".to_string(),
          b"license".to_string(),
      ];

      let values = vector[
          b"{name}".to_string(),
          b"{description}".to_string(),
          b"{tags}".to_string(),
          b"{data_type}".to_string(),
          b"{data_size}".to_string(),
          b"{creator}".to_string(),
          b"{license}".to_string(),
      ];

      let mut d = display::new_with_fields<Dataset>(
          publisher,
          keys,
          values,
          ctx,
      );

      d.update_version();
      d
  }

  public fun get_dataset_name(dataset: &Dataset): String {
    dataset.name
  }

  public fun get_dataset_description(dataset: &Dataset): Option<String> {
    dataset.description
  }

  public fun get_dataset_tags(dataset: &Dataset): Option<vector<String>> {
    dataset.tags
  }

  public fun get_dataset_data_type(dataset: &Dataset): String {
    dataset.data_type
  }

  public fun get_dataset_data_size(dataset: &Dataset): u64 {
    dataset.data_size
  }

  public fun get_dataset_creator(dataset: &Dataset): Option<String> {
    dataset.creator
  }

  public fun get_dataset_license(dataset: &Dataset): String {
    dataset.license
  }

  /// Adds multiple confirmed label annotations to a specific data in dataset
  public fun add_confirmed_label_annotations(dataset: &mut Dataset, path: String, labels: vector<String>, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    while (i < vector::length(&labels)) {
      data.confirmed_label_annotations.push_back(LabelAnnotation { 
        label: *vector::borrow(&labels, i),
        annotated_by: tx_context::sender(ctx)
      });
      i = i + 1;
    }
  }

  /// Clears all pending annotation statistics for a data path (after validation)
  public fun clear_pending_annotation_stats(dataset: &mut Dataset, path: String) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    data.pending_label_stats = vec_map::empty<String, vector<address>>();
    data.pending_bbox_stats = vec_map::empty<vector<u64>, vector<address>>();
    data.pending_skeleton_stats = vec_map::empty<vector<u64>, vector<address>>();
  }
}
