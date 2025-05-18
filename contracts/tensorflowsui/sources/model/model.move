// Copyright (c) OpenGraph, Inc.
// SPDX-License-Identifier: Apache-2.0

/// @title Fully Onchain Neural Network Inference Implementation
module tensorflowsui::model {
    use tensorflowsui::graph::{Self, Graph};
    use tensorflowsui::layer;
    use tensorflowsui::dataset;
    use tensorflowsui::math;
    use std::string::{String};
    use tensorflowsui::tensor;
    use sui::event;
    
    /// @dev Error when weight magnitude and sign vector lengths do not match
    const EWeightsVectorLengthMismatch: u64 = 1008;
    /// @dev Error when bias magnitude and sign vector lengths do not match
    const EBiasesVectorLengthMismatch: u64 = 1009;
    /// @dev Error when scale value is 0
    const EInvalidScale: u64 = 1010;
    /// @dev Error when input vector length does not match first layer input dimension
    const EInputDimensionMismatch: u64 = 1012;
    /// @dev Error when model object is invalid
    const EInvalidModel: u64 = 1013;
    /// @dev Error when model has no graphs
    const EModelHasNoGraphs: u64 = 1014;
    /// @dev Error when layer index is out of bounds
    const ELayerIndexOutOfBounds: u64 = 1015;
    /// @dev Error when dimension index is out of bounds
    const EDimensionIndexOutOfBounds: u64 = 1016;
    /// @dev Error when graph index is out of bounds
    const EGraphIndexOutOfBounds: u64 = 1018;
    /// @dev Error when layer dimensions don't match parameters
    const ELayerDimensionMismatch: u64 = 1020;
    /// @dev Error when model state is invalid for operation
    const EInvalidModelState: u64 = 1021;
    /// @dev Error when partial layer parameters are invalid
    const EInvalidPartialLayerParams: u64 = 1023;

    // Model state constants
    const MODEL_STATE_INCOMPLETE: u8 = 0;
    const MODEL_STATE_COMPLETE: u8 = 1;

    // Layer parameter constants
    const MAX_PARAMS_PER_TRANSACTION: u64 = 3000;

    public struct Model has key {
        id: UID,
        name: String,
        description: String,
        task_type: String,
        graphs: vector<Graph>,
        scale: u64,
        training_dataset_id: Option<ID>,
        test_dataset_ids: Option<vector<ID>>,
        state: u8, // 0 = incomplete, 1 = complete
        current_graph_idx: u64, // tracks which graph is currently being constructed
        current_layer_idx: u64, // tracks which layer is currently being constructed
    }
    
    /// @notice Event emitted when a layer computation is completed
    public struct LayerComputed has copy, drop {
        model_id: address,
        layer_idx: u64,
        output_magnitude: vector<u64>,
        output_sign: vector<u64>,
        activation_type: u64,
    }
    
    /// @notice Event emitted when model prediction is complete
    public struct PredictionCompleted has copy, drop {
        model_id: address,
        output_magnitude: vector<u64>,
        output_sign: vector<u64>,
        argmax_idx: u64,
    }

    /// @notice Event emitted when a model is created
    public struct ModelCreated has copy, drop {
        model_id: address,
        name: String,
        task_type: String,
    }

    /// @notice Event emitted when a model is completed
    public struct ModelCompleted has copy, drop {
        model_id: address,
        graph_count: u64,
        total_layers: u64,
    }

    /// @notice Event emitted when a graph is added to a model
    public struct GraphAdded has copy, drop {
        model_id: address,
        graph_idx: u64,
    }

    /// @notice Event emitted when a layer is added to a graph
    public struct LayerAdded has copy, drop {
        model_id: address,
        graph_idx: u64,
        layer_idx: u64,
        in_dimension: u64,
        out_dimension: u64,
    }

    public struct MODEL has drop {}

    public struct OpenGraphManagerCap has key {
        id: UID
    }

    public fun new_open_graph_manager_cap(_witness: MODEL, ctx: &mut TxContext): OpenGraphManagerCap {
        OpenGraphManagerCap { id: object::new(ctx) }        
    }

    fun init(witness: MODEL, ctx: &mut TxContext) {
        let cap = new_open_graph_manager_cap(witness, ctx);
        transfer::transfer(cap, tx_context::sender(ctx));
    }

    /// @notice Creates an empty model with just metadata (no graphs or layers)
    /// @param name Model name
    /// @param description Model description
    /// @param task_type Model task type (e.g., "classification", "regression")
    /// @param scale Fixed point scale (2^scale)
    /// @param training_dataset_id Training dataset ID (optional)
    /// @param test_dataset_ids List of test dataset IDs (optional)
    /// @param ctx Transaction context
    /// @return New model object ID
    public fun create_model(
        name: String,
        description: String,
        task_type: String,
        scale: u64,
        training_dataset_id: Option<ID>,
        test_dataset_ids: Option<vector<ID>>,
        ctx: &mut TxContext,
    ): Model {
        // Validate scale value
        assert!(scale > 0, EInvalidScale);

        let model = Model {
            id: object::new(ctx),
            name,
            description,
            task_type,
            graphs: vector::empty<Graph>(),
            scale,
            training_dataset_id,
            test_dataset_ids,
            state: MODEL_STATE_INCOMPLETE,
            current_graph_idx: 0,
            current_layer_idx: 0,
        };
        
        // Emit model created event
        event::emit(ModelCreated {
            model_id: object::id_address(&model),
            name,
            task_type,
        });
        
        model
    }

    /// @notice Adds a new empty graph to the model
    /// @param model Model object to add graph to
    public fun add_graph(model: &mut Model) {
        // Model must be in incomplete state
        assert!(model.state == MODEL_STATE_INCOMPLETE, EInvalidModelState);
        
        let graph = graph::new_graph();
        vector::push_back(&mut model.graphs, graph);
        
        let graph_idx = vector::length(&model.graphs) - 1;
        
        // Update current graph index
        model.current_graph_idx = graph_idx;
        model.current_layer_idx = 0;
        
        // Emit graph added event
        event::emit(GraphAdded {
            model_id: object::id_address(model),
            graph_idx,
        });
    }

    /// @notice Starts creating a new layer in the current graph
    /// @param model Model object
    /// @param layer_type Type of layer (e.g., "dense")
    /// @param in_dimension Input dimension of the layer
    /// @param out_dimension Output dimension of the layer
    public fun start_layer(
        model: &mut Model,
        layer_type: String,
        in_dimension: u64,
        out_dimension: u64
    ) {
        // Model must be in incomplete state
        assert!(model.state == MODEL_STATE_INCOMPLETE, EInvalidModelState);
        
        // Check if graph_idx is valid
        let graph_idx = model.current_graph_idx;
        assert!(graph_idx < vector::length(&model.graphs), EGraphIndexOutOfBounds);
        
        // Create empty layer and add to graph
        let empty_weights_mag = vector::empty<u64>();
        let empty_weights_sign = vector::empty<u64>();
        let empty_biases_mag = vector::empty<u64>();
        let empty_biases_sign = vector::empty<u64>();
        
        let layer = layer::new_layer(
            layer_type, 
            in_dimension, 
            out_dimension, 
            empty_weights_mag, 
            empty_weights_sign, 
            empty_biases_mag, 
            empty_biases_sign, 
            model.scale
        );
        
        graph::add_layer(&mut model.graphs[graph_idx], layer);
        
        // Update current layer index
        model.current_layer_idx = graph::get_layer_count(&model.graphs[graph_idx]) - 1;
        
        // Emit layer started event
        event::emit(LayerStarted {
            model_id: object::id_address(model),
            graph_idx,
            layer_idx: model.current_layer_idx,
            in_dimension,
            out_dimension,
        });
    }

    /// @notice Adds partial weights parameters to the current layer
    /// @param model Model object
    /// @param start_idx Starting index in the flattened weights array
    /// @param weights_magnitude Partial weight magnitude values
    /// @param weights_sign Partial weight sign values
    /// @param is_last_chunk Whether this is the last chunk of weights
    public fun add_weights_chunk(
        model: &mut Model,
        start_idx: u64,
        weights_magnitude: vector<u64>,
        weights_sign: vector<u64>,
        is_last_chunk: bool
    ) {
        // Model must be in incomplete state
        assert!(model.state == MODEL_STATE_INCOMPLETE, EInvalidModelState);
        
        // Check if vectors have same length
        assert!(vector::length(&weights_magnitude) == vector::length(&weights_sign), EWeightsVectorLengthMismatch);
        
        // Get current graph and layer
        let graph_idx = model.current_graph_idx;
        let layer_idx = model.current_layer_idx;
        
        assert!(graph_idx < vector::length(&model.graphs), EGraphIndexOutOfBounds);
        assert!(layer_idx < graph::get_layer_count(&model.graphs[graph_idx]), ELayerIndexOutOfBounds);
        
        // Get layer
        let layer = graph::get_layer_at_mut(&mut model.graphs[graph_idx], layer_idx);
        
        // Add weights chunk
        layer::add_weights_chunk(layer, start_idx, weights_magnitude, weights_sign);
        
        // Emit weights chunk added event
        event::emit(WeightsChunkAdded {
            model_id: object::id_address(model),
            graph_idx,
            layer_idx,
            start_idx,
            chunk_size: vector::length(&weights_magnitude),
            is_last_chunk,
        });
    }

    /// @notice Adds partial biases parameters to the current layer
    /// @param model Model object
    /// @param start_idx Starting index in the biases array
    /// @param biases_magnitude Partial bias magnitude values
    /// @param biases_sign Partial bias sign values
    /// @param is_last_chunk Whether this is the last chunk of biases
    public fun add_biases_chunk(
        model: &mut Model,
        start_idx: u64,
        biases_magnitude: vector<u64>,
        biases_sign: vector<u64>,
        is_last_chunk: bool
    ) {
        // Model must be in incomplete state
        assert!(model.state == MODEL_STATE_INCOMPLETE, EInvalidModelState);
        
        // Check if vectors have same length
        assert!(vector::length(&biases_magnitude) == vector::length(&biases_sign), EBiasesVectorLengthMismatch);
        
        // Get current graph and layer
        let graph_idx = model.current_graph_idx;
        let layer_idx = model.current_layer_idx;
        
        assert!(graph_idx < vector::length(&model.graphs), EGraphIndexOutOfBounds);
        assert!(layer_idx < graph::get_layer_count(&model.graphs[graph_idx]), ELayerIndexOutOfBounds);
        
        // Get layer
        let layer = graph::get_layer_at_mut(&mut model.graphs[graph_idx], layer_idx);
        
        // Add biases chunk
        layer::add_biases_chunk(layer, start_idx, biases_magnitude, biases_sign);
        
        // Emit biases chunk added event
        event::emit(BiasesChunkAdded {
            model_id: object::id_address(model),
            graph_idx,
            layer_idx,
            start_idx,
            chunk_size: vector::length(&biases_magnitude),
            is_last_chunk,
        });
    }

    /// @notice Completes the current layer by validating parameter dimensions
    /// @param model Model object
    public fun complete_layer(model: &mut Model, is_final_layer: bool) {
        // Model must be in incomplete state
        assert!(model.state == MODEL_STATE_INCOMPLETE, EInvalidModelState);
        
        // Get current graph and layer
        let graph_idx = model.current_graph_idx;
        let layer_idx = model.current_layer_idx;
        
        assert!(graph_idx < vector::length(&model.graphs), EGraphIndexOutOfBounds);
        assert!(layer_idx < graph::get_layer_count(&model.graphs[graph_idx]), ELayerIndexOutOfBounds);
        
        // Get layer
        let layer = graph::get_layer_at(&model.graphs[graph_idx], layer_idx);
        
        // Validate layer parameters
        let in_dim = layer::get_in_dimension(layer);
        let out_dim = layer::get_out_dimension(layer);
        
        // Get layer tensors
        let weight_tensor = layer::get_weight_tensor(layer);
        let bias_tensor = layer::get_bias_tensor(layer);
        
        // Validate weights and biases dimensions
        let weight_mag = tensor::get_magnitude(weight_tensor);
        let weight_sign = tensor::get_sign(weight_tensor);
        assert!(vector::length(&weight_mag) == in_dim * out_dim, ELayerDimensionMismatch);
        assert!(vector::length(&weight_sign) == in_dim * out_dim, ELayerDimensionMismatch);
        
        let bias_mag = tensor::get_magnitude(bias_tensor);
        let bias_sign = tensor::get_sign(bias_tensor);
        assert!(vector::length(&bias_mag) == out_dim, ELayerDimensionMismatch);
        assert!(vector::length(&bias_sign) == out_dim, ELayerDimensionMismatch);
        
        // Emit layer completed event
        event::emit(LayerCompleted {
            model_id: object::id_address(model),
            graph_idx,
            layer_idx,
            in_dimension: in_dim,
            out_dimension: out_dim,
            is_final_layer,
        });

        if (is_final_layer) {
            model.state = MODEL_STATE_COMPLETE;
        };
    }

    /// @notice Traditional single-transaction layer addition (for layers with parameters under the limit)
    /// @param model Model object
    /// @param graph_idx Index of the graph to add layer to
    /// @param layer_type Type of layer (e.g., "dense")
    /// @param in_dimension Input dimension of the layer
    /// @param out_dimension Output dimension of the layer
    /// @param weights_magnitude Weight magnitude values
    /// @param weights_sign Weight sign values
    /// @param biases_magnitude Bias magnitude values
    /// @param biases_sign Bias sign values
    /// @return Index of the newly added layer
    public fun add_layer(
        model: &mut Model,
        graph_idx: u64,
        layer_type: String,
        in_dimension: u64,
        out_dimension: u64,
        weights_magnitude: vector<u64>,
        weights_sign: vector<u64>,
        biases_magnitude: vector<u64>,
        biases_sign: vector<u64>,
    ) {
        // Check if graph_idx is valid
        assert!(graph_idx < vector::length(&model.graphs), EGraphIndexOutOfBounds);
        
        // Update current indices
        model.current_graph_idx = graph_idx;
        
        // Validate weights and bias vector lengths
        assert!(vector::length(&weights_magnitude) == vector::length(&weights_sign), EWeightsVectorLengthMismatch);
        assert!(vector::length(&biases_magnitude) == vector::length(&biases_sign), EBiasesVectorLengthMismatch);
        assert!(vector::length(&weights_magnitude) == in_dimension * out_dimension, ELayerDimensionMismatch);
        assert!(vector::length(&biases_magnitude) == out_dimension, ELayerDimensionMismatch);
        
        // Verify we're under parameter limit for a single transaction
        let total_params = vector::length(&weights_magnitude) + vector::length(&weights_sign) + vector::length(&biases_magnitude) + vector::length(&biases_sign);
        assert!(total_params <= MAX_PARAMS_PER_TRANSACTION, EInvalidPartialLayerParams);
        
        // Create layer and add to graph
        let layer = layer::new_layer(
            layer_type, 
            in_dimension, 
            out_dimension, 
            weights_magnitude, 
            weights_sign, 
            biases_magnitude, 
            biases_sign, 
            model.scale
        );
        
        graph::add_layer(&mut model.graphs[graph_idx], layer);
        
        // Update current layer index
        model.current_layer_idx = graph::get_layer_count(&model.graphs[graph_idx]) - 1;
        
        // Emit layer added event
        event::emit(LayerAdded {
            model_id: object::id_address(model),
            graph_idx,
            layer_idx: model.current_layer_idx,
            in_dimension,
            out_dimension,
        });
    }

    public fun complete_model(model: Model) {
        assert!(model.state == MODEL_STATE_INCOMPLETE, EInvalidModelState);
        validate_model(&model);

        event::emit(ModelCompleted {
            model_id: object::id_address(&model),
            graph_count: vector::length(&model.graphs),
            total_layers: get_total_layers(&model),
        });

        transfer::share_object(model);
    }

    public fun delete_model(model: Model, _: &OpenGraphManagerCap) {
        let Model { id, .. } = model;
        id.delete();
    }

    /// @notice Helper function to get model name as String
    /// @param model Model object
    /// @return Name of the model
    public fun get_name(model: &Model): &String {
        &model.name
    }

    /// @notice Helper function to get model description as String
    /// @param model Model object
    /// @return Description of the model
    public fun get_description(model: &Model): &String {
        &model.description
    }

    /// @notice Helper function to get model task type as String
    /// @param model Model object
    /// @return Task type of the model (e.g., "classification", "regression")
    public fun get_task_type(model: &Model): &String {
        &model.task_type
    }

    /// @notice Helper function to get model scale
    /// @param model Model object
    /// @return Scale value used for fixed-point calculations
    public fun get_scale(model: &Model): u64 {
        model.scale
    }

    /// @notice Calculate the total number of layers in the model
    /// @param model Model object
    /// @return Total number of layers in the model
    fun get_total_layers(model: &Model): u64 {
        let graph_count = vector::length(&model.graphs);
        let mut total = 0;
        
        let mut i = 0;
        while (i < graph_count) {
            total = total + graph::get_layer_count(&model.graphs[i]);
            i = i + 1;
        };
        
        total
    }


    /// Adds a test dataset to the model.
    public fun add_test_dataset(model: &mut Model, test_dataset: &dataset::Dataset) {
        if (option::is_none(&model.test_dataset_ids)) {
            model.test_dataset_ids = option::some(vector::empty<ID>());
        };
        vector::push_back(option::borrow_mut(&mut model.test_dataset_ids), object::id(test_dataset));
    }

    /// Removes a test dataset from the model.
    /// Returns true if the dataset was found and removed, false otherwise.
    public fun remove_test_dataset(model: &mut Model, test_dataset_id: ID): bool {
        if (option::is_none(&model.test_dataset_ids)) {
            return false
        };
        
        let mut i = 0;
        let len = vector::length(option::borrow(&model.test_dataset_ids));
        while (i < len) {
            let current_id = vector::borrow(option::borrow(&model.test_dataset_ids), i);
            if (*current_id == test_dataset_id) {
                vector::remove(option::borrow_mut(&mut model.test_dataset_ids), i);
                return true
            };
            i = i + 1;
        };
        false
    }

    /// Gets the training dataset ID.
    public fun get_training_dataset_id(model: &Model): Option<ID> {
        model.training_dataset_id
    }

    /// Gets all test dataset IDs.
    public fun get_test_dataset_ids(model: &Model): &Option<vector<ID>> {
        &model.test_dataset_ids
    }

    /// Gets the number of test datasets.
    public fun get_test_dataset_count(model: &Model): u64 {
        if (option::is_none(&model.test_dataset_ids)) {
            return 0
        };
        vector::length(option::borrow(&model.test_dataset_ids))
    }

    /// @notice Run inference on the model with provided input
    /// @param model Model object to run inference on
    /// @param input_magnitude Magnitude values of the input vector
    /// @param input_sign Sign values of the input vector (0 for positive, 1 for negative)
    /// @return Tuple of (magnitude vector, sign vector, argmax index) of the model output
    entry public fun predict(
        model: &Model,
        input_magnitude: vector<u64>,
        input_sign: vector<u64>
    ): (vector<u64>, vector<u64>, u64) {
        // Validate model has at least one graph
        assert!(vector::length(&model.graphs) > 0, EModelHasNoGraphs);
        
        // Get the first graph (currently we only support one graph per model)
        let graph = vector::borrow(&model.graphs, 0);
        
        // Get first layer to validate input dimensions
        assert!(graph::get_layer_count(graph) > 0, EInvalidModel);
        let first_layer = graph::get_layer_at(graph, 0);
        let input_dim = layer::get_in_dimension(first_layer);
        
        // Validate input dimensions
        assert!(vector::length(&input_magnitude) == input_dim, EInputDimensionMismatch);
        assert!(vector::length(&input_sign) == input_dim, EInputDimensionMismatch);
        
        // Create input tensor (batch size 1)
        let input_shape = vector[1, input_dim];
        let input_tensor = tensor::new_tensor(
            input_shape,
            input_magnitude,
            input_sign,
            model.scale
        );
        
        // Process through all layers in the graph
        let mut current_tensor = input_tensor;
        let layer_count = graph::get_layer_count(graph);
        
        let mut i = 0;
        while (i < layer_count) {
            let layer = graph::get_layer_at(graph, i);
            let weight_tensor = layer::get_weight_tensor(layer);
            let bias_tensor = layer::get_bias_tensor(layer);
            
            // Apply activation function (ReLU for all layers except the last one)
            let activation_type = if (i == layer_count - 1) { 0 } else { 1 }; // 0=None, 1=ReLU
            
            // Apply layer computation
            // TODO: select computation function based on layer type (dense, conv, etc.)
            current_tensor = layer::compute_dense_layer(
                &current_tensor,
                weight_tensor,
                bias_tensor,
                activation_type
            );
            
            i = i + 1;
        };
        
        // Extract results from the final tensor
        let result_mag = tensor::get_magnitude(&current_tensor);
        let result_sign = tensor::get_sign(&current_tensor);
        
        // Find argmax if we have results
        let max_idx = find_argmax(&result_mag, &result_sign);
        
        // Emit prediction completed event
        event::emit(PredictionCompleted {
            model_id: object::id_address(model),
            output_magnitude: result_mag,
            output_sign: result_sign,
            argmax_idx: max_idx,
        });
        
        (result_mag, result_sign, max_idx)
    }
    
    /// @notice Helper function to find the argmax index in result vectors
    fun find_argmax(magnitudes: &vector<u64>, signs: &vector<u64>): u64 {
        let mut max_idx = 0;
        let mut max_val = 0;
        let result_len = vector::length(magnitudes);
        
        if (result_len > 0) {
            let mut j = 0;
            while (j < result_len) {
                let val = vector::borrow(magnitudes, j);
                let sign = vector::borrow(signs, j);
                
                // Only consider positive values or zero for argmax
                if (*sign == 0 && *val > max_val) {
                    max_val = *val;
                    max_idx = j;
                };
                
                j = j + 1;
            };
        };
        
        max_idx
    }
    
    /// @notice Process a single layer and emit result as event (gas efficient version)
    /// @param model Model object to run inference on
    /// @param layer_idx Index of the layer to process
    /// @param input_magnitude Magnitude values of the input vector
    /// @param input_sign Sign values of the input vector
    /// @return Tuple of (magnitude vector, sign vector, optional argmax index for final layer)
    entry public fun predict_layer(
        model: &Model,
        layer_idx: u64,
        input_magnitude: vector<u64>,
        input_sign: vector<u64>
    ): (vector<u64>, vector<u64>, Option<u64>) {
        // Validate model has at least one graph
        assert!(vector::length(&model.graphs) > 0, EModelHasNoGraphs);
        
        // Get the first graph (currently we only support one graph per model)
        let graph = vector::borrow(&model.graphs, 0);
        
        // Check if layer_idx is valid
        let layer_count = graph::get_layer_count(graph);
        assert!(layer_idx < layer_count, ELayerIndexOutOfBounds);
        
        // Check if this is the last layer
        let is_last_layer = layer_idx == layer_count - 1;
        
        // Get the target layer
        let layer = graph::get_layer_at(graph, layer_idx);
        let input_dim = layer::get_in_dimension(layer);
        
        // Validate input dimensions
        assert!(vector::length(&input_magnitude) == input_dim, EInputDimensionMismatch);
        assert!(vector::length(&input_sign) == input_dim, EInputDimensionMismatch);
        
        // Create input tensor (batch size 1)
        let input_shape = vector[1, input_dim];
        let input_tensor = tensor::new_tensor(
            input_shape,
            input_magnitude,
            input_sign,
            model.scale
        );
        
        // Get layer tensors
        let weight_tensor = layer::get_weight_tensor(layer);
        let bias_tensor = layer::get_bias_tensor(layer);
        
        // Apply activation function (ReLU for all layers except the last one)
        let activation_type = if (is_last_layer) { 0 } else { 1 }; // 0=None, 1=ReLU
        
        // Compute dense layer
        let result_tensor = layer::compute_dense_layer(
            &input_tensor,
            weight_tensor,
            bias_tensor,
            activation_type
        );
        
        // Extract results from the layer output tensor
        let result_mag = tensor::get_magnitude(&result_tensor);
        let result_sign = tensor::get_sign(&result_tensor);
        
        // For the last layer, calculate the argmax
        let mut argmax_idx = option::none();
        
        if (is_last_layer) {
            // Find argmax if we have results
            let max_idx = find_argmax(&result_mag, &result_sign);
            
            // Emit prediction completed event
            event::emit(PredictionCompleted {
                model_id: object::id_address(model),
                output_magnitude: result_mag,
                output_sign: result_sign,
                argmax_idx: max_idx,
            });
            
            argmax_idx = option::some(max_idx);
        };

        // Emit layer computed event
        event::emit(LayerComputed {
            model_id: object::id_address(model),
            layer_idx,
            output_magnitude: result_mag,
            output_sign: result_sign,
            activation_type,
        });
        
        (result_mag, result_sign, argmax_idx)
    }

    /// @notice Process a single output dimension of a layer (gas efficient version)
    /// @param model Model object to run inference on
    /// @param layer_idx Index of the layer to process
    /// @param output_dim_idx Index of the output dimension to process (0 to out_dim-1)
    /// @param input_magnitude Magnitude values of the input vector
    /// @param input_sign Sign values of the input vector
    /// @param result_magnitudes Vector of accumulated magnitude values
    /// @param result_signs Vector of accumulated sign values
    /// @return Tuple of (output magnitude scalar, output sign scalar, output dimension index, is last dimension)
    entry public fun predict_layer_partial(
        model: &Model,
        layer_idx: u64,
        output_dim_idx: u64,
        input_magnitude: vector<u64>,
        input_sign: vector<u64>,
        mut result_magnitudes: vector<u64>,
        mut result_signs: vector<u64>,
    ): (vector<u64>, vector<u64>, u64, bool) {
        // Validate model has at least one graph
        assert!(vector::length(&model.graphs) > 0, EModelHasNoGraphs);
        
        // Get the first graph (currently we only support one graph per model)
        let graph = vector::borrow(&model.graphs, 0);
        
        // Check if layer_idx is valid
        let layer_count = graph::get_layer_count(graph);
        assert!(layer_idx < layer_count, ELayerIndexOutOfBounds);
        
        // Get the target layer
        let layer = graph::get_layer_at(graph, layer_idx);
        let input_dim = layer::get_in_dimension(layer);
        let output_dim = layer::get_out_dimension(layer);
        
        // Validate output dimension index
        assert!(output_dim_idx < output_dim, EDimensionIndexOutOfBounds);
        
        // Validate input dimensions
        assert!(vector::length(&input_magnitude) == input_dim, EInputDimensionMismatch);
        assert!(vector::length(&input_sign) == input_dim, EInputDimensionMismatch);
        
        // Check if this is the last layer and last dimension
        let is_last_layer = layer_idx == layer_count - 1;
        let is_last_dimension = output_dim_idx == output_dim - 1;
        
        // Get weight and bias tensors
        let weight_tensor = layer::get_weight_tensor(layer);
        let bias_tensor = layer::get_bias_tensor(layer);
        
        // Extract weight and bias data
        let weight_mag = tensor::get_magnitude(weight_tensor);
        let weight_sign = tensor::get_sign(weight_tensor);
        let bias_mag = tensor::get_magnitude(bias_tensor);
        let bias_sign = tensor::get_sign(bias_tensor);
        
        // Calculate single output dimension (dot product for this dimension only)
        let mut result_mag = 0;
        let mut result_sign = 0;
        
        // Add bias for this dimension
        if (output_dim_idx < vector::length(&bias_mag)) {
            result_mag = *vector::borrow(&bias_mag, output_dim_idx);
            result_sign = *vector::borrow(&bias_sign, output_dim_idx);
        };
        
        // Calculate dot product for this single output dimension
        let mut i = 0;
        while (i < input_dim) {
            // Get weight for this connection (input_dim x output_dim_idx)
            // Flattened index calculation for weight matrix
            let weight_idx = i * output_dim + output_dim_idx;
            
            if (weight_idx < vector::length(&weight_mag)) {
                let weight_mag_val = *vector::borrow(&weight_mag, weight_idx);
                let weight_sign_val = *vector::borrow(&weight_sign, weight_idx);
                
                // Get input value
                let input_mag_val = *vector::borrow(&input_magnitude, i);
                let input_sign_val = *vector::borrow(&input_sign, i);
                
                // Multiply
                let product_mag = input_mag_val * weight_mag_val;
                let product_sign = input_sign_val ^ weight_sign_val; // XOR for sign multiplication
                
                // Apply scaling after multiplication
                let scaled_product_mag = math::scale_up(product_mag, model.scale);
                
                // Add to result (considering signs)
                if (result_sign == product_sign) {
                    // Same sign, simply add magnitudes
                    result_mag = result_mag + scaled_product_mag;
                } else {
                    // Different signs, subtract smaller from larger and determine sign
                    if (result_mag > scaled_product_mag) {
                        result_mag = result_mag - scaled_product_mag;
                        // result_sign stays the same
                    } else if (result_mag < scaled_product_mag) {
                        result_mag = scaled_product_mag - result_mag;
                        result_sign = product_sign; // Take sign of the larger value
                    } else {
                        // Equal magnitudes with different signs cancel out
                        result_mag = 0;
                        result_sign = 0; // Default to positive for zero
                    }
                };
            };
            
            i = i + 1;
        };
        
        // Apply activation if not last layer (ReLU: max(0, x))
        if (!is_last_layer && result_sign == 1) {
            // If negative and using ReLU, set to zero
            result_mag = 0;
            result_sign = 0;
        };

        vector::push_back(&mut result_magnitudes, result_mag);
        vector::push_back(&mut result_signs, result_sign);
        
        // Emit partial result event
        event::emit(LayerPartialComputed {
            model_id: object::id_address(model),
            layer_idx,
            output_dim_idx,
            output_magnitude: result_mag,
            output_sign: result_sign,
            is_last_dimension
        });
        
        // If this is the last layer and last dimension, we can calculate the argmax across collected results
        if (is_last_layer && is_last_dimension) {
            // Calculate argmax from the accumulated result vectors
            let argmax_idx = find_argmax(&result_magnitudes, &result_signs);
            
            // Emit completion event with the full accumulated results
            event::emit(PredictionCompleted {
                model_id: object::id_address(model),
                output_magnitude: result_magnitudes,
                output_sign: result_signs,
                argmax_idx
            });
        };
        
        (result_magnitudes, result_signs, output_dim_idx, is_last_dimension)
    }
    
    /// @notice Event emitted when a partial layer computation is completed
    public struct LayerPartialComputed has copy, drop {
        model_id: address,
        layer_idx: u64,
        output_dim_idx: u64,
        output_magnitude: u64,
        output_sign: u64,
        is_last_dimension: bool
    }

    /// @notice Validates the model structure before completion
    /// @param model Model to validate
    fun validate_model(model: &Model) {
        // Check if model has at least one graph
        assert!(vector::length(&model.graphs) > 0, EModelHasNoGraphs);
        
        let graph_count = vector::length(&model.graphs);
        let mut graph_idx = 0;
        
        while (graph_idx < graph_count) {
            let graph = vector::borrow(&model.graphs, graph_idx);
            
            // Check that graph has at least one layer
            let layer_count = graph::get_layer_count(graph);
            assert!(layer_count > 0, EInvalidModel);
            
            // Validate layer connections (output dim of layer i = input dim of layer i+1)
            let mut layer_idx = 0;
            while (layer_idx < layer_count - 1) {
                let current_layer = graph::get_layer_at(graph, layer_idx);
                let next_layer = graph::get_layer_at(graph, layer_idx + 1);
                
                assert!(
                    layer::get_out_dimension(current_layer) == layer::get_in_dimension(next_layer),
                    ELayerDimensionMismatch
                );
                
                layer_idx = layer_idx + 1;
            };
            
            // Validate all layers' parameters
            layer_idx = 0;
            while (layer_idx < layer_count) {
                let layer = graph::get_layer_at(graph, layer_idx);
                
                let in_dim = layer::get_in_dimension(layer);
                let out_dim = layer::get_out_dimension(layer);
                
                let weight_tensor = layer::get_weight_tensor(layer);
                let bias_tensor = layer::get_bias_tensor(layer);
                
                // Validate tensor dimensions
                let weight_mag = tensor::get_magnitude(weight_tensor);
                let weight_sign = tensor::get_sign(weight_tensor);
                let bias_mag = tensor::get_magnitude(bias_tensor);
                let bias_sign = tensor::get_sign(bias_tensor);
                
                assert!(
                    vector::length(&weight_mag) == in_dim * out_dim,
                    ELayerDimensionMismatch
                );
                assert!(
                    vector::length(&weight_sign) == in_dim * out_dim,
                    ELayerDimensionMismatch
                );
                assert!(
                    vector::length(&bias_mag) == out_dim,
                    ELayerDimensionMismatch
                );
                assert!(
                    vector::length(&bias_sign) == out_dim,
                    ELayerDimensionMismatch
                );
                
                layer_idx = layer_idx + 1;
            };
            
            graph_idx = graph_idx + 1;
        }
    }

    /// @notice Event emitted when a layer creation starts
    public struct LayerStarted has copy, drop {
        model_id: address,
        graph_idx: u64,
        layer_idx: u64,
        in_dimension: u64,
        out_dimension: u64,
    }

    /// @notice Event emitted when a layer is completed
    public struct LayerCompleted has copy, drop {
        model_id: address,
        graph_idx: u64,
        layer_idx: u64,
        in_dimension: u64,
        out_dimension: u64,
        is_final_layer: bool,
    }

    /// @notice Event emitted when a weights chunk is added
    public struct WeightsChunkAdded has copy, drop {
        model_id: address,
        graph_idx: u64,
        layer_idx: u64,
        start_idx: u64,
        chunk_size: u64,
        is_last_chunk: bool,
    }

    /// @notice Event emitted when a biases chunk is added
    public struct BiasesChunkAdded has copy, drop {
        model_id: address,
        graph_idx: u64,
        layer_idx: u64,
        start_idx: u64,
        chunk_size: u64,
        is_last_chunk: bool,
    }

}