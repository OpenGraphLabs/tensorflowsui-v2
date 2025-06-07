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

  /// Status of an annotation
  public struct AnnotationStatus has copy, drop, store {
    is_confirmed: bool,
    confirmed_at: Option<u64>,  // timestamp when confirmed
    confirmed_by: Option<address>,  // address of the confirmer
  }

  /// An annotation type for a data in a dataset.
  public struct LabelAnnotation has copy, drop, store {
    // label of the annotation
    label: String,
    // address of the user who created this annotation
    annotated_by: address,
    // status of the annotation
    status: AnnotationStatus,
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
    // status of the annotation
    status: AnnotationStatus,
  }

  /// A skeleton annotation for a data in a dataset.
  public struct SkeletonAnnotation has copy, drop, store {
    // keypoints of the skeleton [(x1,y1), (x2,y2), ...]
    keypoints: vector<Point>,
    // edges connecting keypoints [(start_idx, end_idx), ...]
    edges: vector<Edge>,
    // address of the user who created this annotation
    annotated_by: address,
    // status of the annotation
    status: AnnotationStatus,
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
      pending_label_stats: VecMap<String, u64>,

      // All annotations
      label_annotations: vector<LabelAnnotation>,
      bbox_annotations: vector<BBoxAnnotation>,
      skeleton_annotations: vector<SkeletonAnnotation>,
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
          pending_label_stats: vec_map::empty<String, u64>(),
          label_annotations: vector[],
          bbox_annotations: vector[],
          skeleton_annotations: vector[],
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

  /// Creates a new pending annotation status
  public fun new_pending_status(): AnnotationStatus {
    AnnotationStatus {
      is_confirmed: false,
      confirmed_at: option::none(),
      confirmed_by: option::none(),
    }
  }

  /// Updates an existing annotation status to confirmed state
  public fun update_to_confirmed_status(status: &mut AnnotationStatus, confirmed_by: address, ctx: &mut TxContext) {
    status.is_confirmed = true;
    status.confirmed_at = option::some(tx_context::epoch(ctx));
    status.confirmed_by = option::some(confirmed_by);
  }

  /// Adds a label annotation (initially in pending state)
  public fun add_label_annotation(dataset: &mut Dataset, path: String, label: String, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let sender = tx_context::sender(ctx);
    
    // Update pending stats
    if (vec_map::contains(&data.pending_label_stats, &label)) {
      let count = vec_map::get_mut(&mut data.pending_label_stats, &label);
      *count = *count + 1;
    } else {
      vec_map::insert(&mut data.pending_label_stats, label, 1);
    };

    // Create and add the annotation
    let annotation = LabelAnnotation {
      label,
      annotated_by: sender,
      status: new_pending_status(),
    };
    vector::push_back(&mut data.label_annotations, annotation);
  }

  /// Adds a bounding box annotation (initially in pending state)
  public fun add_bbox_annotation(dataset: &mut Dataset, path: String, x1: u64, y1: u64, x2: u64, y2: u64, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let sender = tx_context::sender(ctx);
    
    // Create and add the annotation
    let annotation = BBoxAnnotation {
      x1, y1, x2, y2,
      annotated_by: sender,
      status: new_pending_status(),
    };
    vector::push_back(&mut data.bbox_annotations, annotation);
  }

  /// Adds a skeleton annotation (initially in pending state)
  public fun add_skeleton_annotation(
    dataset: &mut Dataset, 
    path: String, 
    keypoints: vector<Point>,
    edges: vector<Edge>,
    ctx: &mut TxContext
  ) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let sender = tx_context::sender(ctx);
    
    // Create and add the annotation
    let annotation = SkeletonAnnotation {
      keypoints,
      edges,
      annotated_by: sender,
      status: new_pending_status(),
    };
    vector::push_back(&mut data.skeleton_annotations, annotation);
  }

  /// Adds label annotations in batch
  public fun batch_add_label_annotations(
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
      add_label_annotation(dataset, path, label, ctx);
      i = i + 1;
    }
  }

  /// Adds bounding box annotations in batch
  public fun batch_add_bbox_annotations(
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
      add_bbox_annotation(
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

  /// Adds skeleton annotations in batch
  public fun batch_add_skeleton_annotations(
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
      add_skeleton_annotation(
        dataset,
        path,
        *vector::borrow(&keypoints, i),
        *vector::borrow(&edges, i),
        ctx
      );
      i = i + 1;
    }
  }


  /// Confirms label annotations
  public fun confirm_label_annotations(dataset: &mut Dataset, path: String, indices: vector<u64>, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    while (i < vector::length(&indices)) {
              let idx = *vector::borrow(&indices, i);
        let annotation = vector::borrow_mut(&mut data.label_annotations, idx);
        update_to_confirmed_status(&mut annotation.status, tx_context::sender(ctx), ctx);
        i = i + 1;
    }
  }

  /// Confirms bounding box annotations
  public fun confirm_bbox_annotations(dataset: &mut Dataset, path: String, indices: vector<u64>, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    while (i < vector::length(&indices)) {
              let idx = *vector::borrow(&indices, i);
        let annotation = vector::borrow_mut(&mut data.bbox_annotations, idx);
        update_to_confirmed_status(&mut annotation.status, tx_context::sender(ctx), ctx);
        i = i + 1;
    }
  }

  /// Confirms skeleton annotations
  public fun confirm_skeleton_annotations(dataset: &mut Dataset, path: String, indices: vector<u64>, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    while (i < vector::length(&indices)) {
              let idx = *vector::borrow(&indices, i);
        let annotation = vector::borrow_mut(&mut data.skeleton_annotations, idx);
        update_to_confirmed_status(&mut annotation.status, tx_context::sender(ctx), ctx);
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

  /// Validates pending label annotations and confirms them if they meet the validation criteria
  public fun validate_label_annotations(dataset: &mut Dataset, path: String, label: String, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let sender = tx_context::sender(ctx);
    
    // Check if there are any pending annotations for this label
    assert!(vec_map::contains(&data.pending_label_stats, &label), ERROR_NO_PENDING_ANNOTATIONS);
    
    // Get the count for this label
    let count = *vec_map::get(&data.pending_label_stats, &label);
    
    // Require at least 2 annotations for validation
    assert!(count >= 2, ERROR_INSUFFICIENT_ANNOTATIONS);
    
    // Find and confirm all matching label annotations
    let mut i = 0;
    while (i < vector::length(&mut data.label_annotations)) {
      let annotation = vector::borrow_mut(&mut data.label_annotations, i);
      if (!annotation.status.is_confirmed && annotation.label == label) {
        update_to_confirmed_status(&mut annotation.status, sender, ctx);
      };
      i = i + 1;
    };
    
    // Remove the validated annotations from pending stats
    vec_map::remove(&mut data.pending_label_stats, &label);
  }

  /// Clears all pending annotation statistics for a data path (after validation)
  public fun clear_pending_annotation_stats(dataset: &mut Dataset, path: String) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    data.pending_label_stats = vec_map::empty<String, u64>();
  }

  /// Validates pending bounding box annotations and confirms them if they meet the validation criteria
  public fun validate_bbox_annotations(dataset: &mut Dataset, path: String, x1: u64, y1: u64, x2: u64, y2: u64, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let sender = tx_context::sender(ctx);
    
    // Find and confirm all matching bbox annotations
    let mut i = 0;
    while (i < vector::length(&mut data.bbox_annotations)) {
      let annotation = vector::borrow_mut(&mut data.bbox_annotations, i);
      if (!annotation.status.is_confirmed && 
        annotation.x1 == x1 && 
        annotation.y1 == y1 && 
        annotation.x2 == x2 && 
        annotation.y2 == y2) {
        update_to_confirmed_status(&mut annotation.status, sender, ctx);
      };
      i = i + 1;
    };
  }

  /// Validates pending skeleton annotations and confirms them if they meet the validation criteria
  public fun validate_skeleton_annotations(
    dataset: &mut Dataset, 
    path: String, 
    keypoints: vector<Point>,
    edges: vector<Edge>,
    ctx: &mut TxContext
  ) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let sender = tx_context::sender(ctx);
    
    // Find and confirm all matching skeleton annotations
    let mut i = 0;
    while (i < vector::length(&mut data.skeleton_annotations)) {
      let annotation = vector::borrow_mut(&mut data.skeleton_annotations, i);
      if (!annotation.status.is_confirmed && 
        compare_points(&annotation.keypoints, &keypoints) && 
        compare_edges(&annotation.edges, &edges)) {
        update_to_confirmed_status(&mut annotation.status, sender, ctx);
      };
      i = i + 1;
    };
  }

  /// Helper function to compare two vectors of Points
  fun compare_points(points1: &vector<Point>, points2: &vector<Point>): bool {
    if (vector::length(points1) != vector::length(points2)) {
      return false
    };
    
    let mut i = 0;
    while (i < vector::length(points1)) {
      let p1 = vector::borrow(points1, i);
      let p2 = vector::borrow(points2, i);
      if (p1.x != p2.x || p1.y != p2.y) {
        return false
      };
      i = i + 1;
    };
    true
  }

  /// Helper function to compare two vectors of Edges
  fun compare_edges(edges1: &vector<Edge>, edges2: &vector<Edge>): bool {
    if (vector::length(edges1) != vector::length(edges2)) {
      return false
    };
    
    let mut i = 0;
    while (i < vector::length(edges1)) {
      let e1 = vector::borrow(edges1, i);
      let e2 = vector::borrow(edges2, i);
      if (e1.start_idx != e2.start_idx || e1.end_idx != e2.end_idx) {
        return false
      };
      i = i + 1;
    };
    true
  }

  // Error constants
  const ERROR_NO_PENDING_ANNOTATIONS: u64 = 1;
  const ERROR_INSUFFICIENT_ANNOTATIONS: u64 = 2;

  /// Validates a batch of label annotations
  public fun validate_label_annotations_batch(dataset: &mut Dataset, path: String, indices: vector<u64>, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    let sender = tx_context::sender(ctx);
    while (i < vector::length(&indices)) {
      let idx = *vector::borrow(&indices, i);
      let annotation = vector::borrow_mut(&mut data.label_annotations, idx);
      update_to_confirmed_status(&mut annotation.status, sender, ctx);
      i = i + 1;
    }
  }

  /// Validates a batch of bounding box annotations
  public fun validate_bbox_annotations_batch(dataset: &mut Dataset, path: String, indices: vector<u64>, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    let sender = tx_context::sender(ctx);
    while (i < vector::length(&indices)) {
      let idx = *vector::borrow(&indices, i);
      let annotation = vector::borrow_mut(&mut data.bbox_annotations, idx);
      update_to_confirmed_status(&mut annotation.status, sender, ctx);
      i = i + 1;
    }
  }

  /// Validates a batch of skeleton annotations
  public fun validate_skeleton_annotations_batch(dataset: &mut Dataset, path: String, indices: vector<u64>, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    let sender = tx_context::sender(ctx);
    while (i < vector::length(&indices)) {
      let idx = *vector::borrow(&indices, i);
      let annotation = vector::borrow_mut(&mut data.skeleton_annotations, idx);
      update_to_confirmed_status(&mut annotation.status, sender, ctx);
      i = i + 1;
    }
  }

  /// Validates all pending label annotations for a given label
  public fun validate_all_pending_label_annotations(dataset: &mut Dataset, path: String, labels: vector<String>, ctx: &mut TxContext) {
    let data = dynamic_field::borrow_mut<DataPath, Data>(&mut dataset.id, new_data_path(path));
    let mut i = 0;
    let sender = tx_context::sender(ctx);
    while (i < vector::length(&labels)) {
      let label = vector::borrow(&labels, i);
      let mut j = 0;
      while (j < vector::length(&mut data.label_annotations)) {
        let annotation = vector::borrow_mut(&mut data.label_annotations, j);
        if (!annotation.status.is_confirmed && annotation.label == *label) {
          update_to_confirmed_status(&mut annotation.status, sender, ctx);
        };
        j = j + 1;
      };
      i = i + 1;
    }
  }
}
