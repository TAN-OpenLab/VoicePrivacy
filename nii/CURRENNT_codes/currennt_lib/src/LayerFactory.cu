/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "LayerFactory.hpp"

#include "layers/InputLayer.hpp"
#include "layers/FeedForwardLayer.hpp"
#include "layers/SoftmaxLayer.hpp"
#include "layers/LstmLayer.hpp"
#include "layers/SsePostOutputLayer.hpp"
#include "layers/KLPostOutputLayer.hpp"
#include "layers/RmsePostOutputLayer.hpp"
#include "layers/CePostOutputLayer.hpp"
#include "layers/SseMaskPostOutputLayer.hpp"
#include "layers/WeightedSsePostOutputLayer.hpp"
#include "layers/BinaryClassificationLayer.hpp"
#include "layers/MulticlassClassificationLayer.hpp"
#include "layers/MiddleOutputLayer.hpp"
#include "layers/BatchNorm.hpp"
#include "layers/OperationLayer.hpp"
#include "layers/FeatMatch.hpp"
#include "layers/vaeMiddleLayer.hpp"
#include "activation_functions/Tanh.cuh"
#include "activation_functions/Logistic.cuh"
#include "activation_functions/Identity.cuh"
#include "activation_functions/Relu.cuh"

#include "layers/SkipAddLayer.hpp"
#include "layers/SkipCatLayer.hpp"
#include "layers/SkipParaLayer.hpp"
#include "layers/MDNLayer.hpp"
#include "layers/CNNLayer.hpp"
#include "layers/Maxpooling.hpp"
#include "layers/RnnLayer.hpp"
#include "layers/ParaLayer.hpp"
#include "layers/FeedBackLayer.hpp"
#include "layers/wavNetCore.hpp"
#include "layers/ExternalLoader.hpp"
#include "layers/vqLayer.hpp"
#include "layers/NormFlowLayer.hpp"
#include "layers/StrucTransformLayer.hpp"
#include "layers/SignalGenLayer.hpp"
#include "layers/DistillingLayer.hpp"
#include "layers/embedding.hpp"
#include "layers/DFTErrorPostoutputLayer.hpp"
#include "layers/SkipWeightAddLayer.hpp"
#include "layers/FilteringLayer.hpp"
#include "layers/SkipForFeatTransform.hpp"
#include "layers/SkipMergeDim.hpp"
#include "layers/SsePostOutputLayerForFeatTransform.hpp"
#include "layers/FeatExtract.hpp"
#include "layers/GatedActLayer.hpp"
#include "layers/FeedBackHiddenLayer.hpp"
#include "layers/SsePostOutputLayerMultiSource.hpp"
#include "layers/SincFilterLayer.hpp"
#include "layers/SelfAttention.hpp"
#include "layers/InterMetricSSE.hpp"
#include "layers/InterMetricSoftmax.hpp"
#include "layers/RandomShuffleLayer.hpp"
#include "layers/SseCosPostOutputLayer.hpp"
#include "layers/InterWeaveLayer.hpp"
#include "layers/MutualInforPostOutputLayer.hpp"
#include "layers/LayerNorm.hpp"
#include "layers/BatchNormNew.hpp"
#include "layers/SpecialFeedBackLayer.hpp"
#include <stdexcept>


template <typename TDevice>
layers::Layer<TDevice>* LayerFactory<TDevice>::createLayer(
		const std::string &layerType,        
		const helpers::JsonValue &layerChild,
		const helpers::JsonValue &weightsSection, 
		int parallelSequences,
		int maxSeqLength,
		int layerID,
		layers::Layer<TDevice> *precedingLayer
		)
{
    using namespace layers;
    using namespace activation_functions;

    if (layerType == "input")
    	return new InputLayer<TDevice>(layerChild, parallelSequences, maxSeqLength, layerID);
    else if (layerType == "feedforward_tanh")
    	return new FeedForwardLayer<TDevice, Tanh>(layerChild, weightsSection,
						   *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "feedforward_logistic")
    	return new FeedForwardLayer<TDevice, Logistic>(layerChild, weightsSection,
						       *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "feedforward_identity")
    	return new FeedForwardLayer<TDevice, Identity>(layerChild, weightsSection,
						       *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "feedforward_relu")
    	return new FeedForwardLayer<TDevice, Relu>(layerChild, weightsSection,
						   *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "paralayer")
    	return new ParaLayer<TDevice, Identity>(layerChild, weightsSection,
						*precedingLayer, maxSeqLength, layerID);    
    else if (layerType == "softmax")
    	return new SoftmaxLayer<TDevice, Identity>(layerChild, weightsSection,
						   *precedingLayer, maxSeqLength, layerID, false);
    else if (layerType == "feedforward_softmax")
    	return new SoftmaxLayer<TDevice, Identity>(layerChild, weightsSection,
						   *precedingLayer, maxSeqLength, layerID, true);
    else if (layerType == "lstm")
    	return new LstmLayer<TDevice>(layerChild, weightsSection,
				      *precedingLayer, maxSeqLength, layerID, false);
    else if (layerType == "blstm")
    	return new LstmLayer<TDevice>(layerChild, weightsSection,
				      *precedingLayer, maxSeqLength, layerID, true);
    else if (layerType == "rnn")
    	return new RnnLayer<TDevice>(layerChild, weightsSection,
				     *precedingLayer, maxSeqLength, layerID, false);
    else if (layerType == "brnn")
    	return new RnnLayer<TDevice>(layerChild, weightsSection,
				     *precedingLayer, maxSeqLength, layerID, true);
    else if (layerType == "feedback")
    	return new FeedBackLayer<TDevice>(layerChild, weightsSection,
					  *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "batchnorm")
    	return new BatchNormLayer<TDevice>(layerChild, weightsSection,
					   *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "cnn")
        return new CNNLayer<TDevice>(layerChild, weightsSection,
				     *precedingLayer, maxSeqLength, layerID);    
    else if (layerType == "maxpooling")
        return new MaxPoolingLayer<TDevice>(layerChild, weightsSection,
					    *precedingLayer, maxSeqLength, layerID);    
    else if (layerType == "middleoutput")
        return new MiddleOutputLayer<TDevice>(layerChild, *precedingLayer, maxSeqLength, layerID);
    
    else if (layerType == "operator")
        return new OperationLayer<TDevice>(layerChild, weightsSection,
					   *precedingLayer, maxSeqLength, layerID);    
    else if (layerType == "featmatch")
        return new FeatMatchLayer<TDevice>(layerChild, *precedingLayer, maxSeqLength, layerID);
    
    else if (layerType == "vae")
        return new VaeMiddleLayer<TDevice>(layerChild, weightsSection,
					   *precedingLayer, maxSeqLength, layerID);
    
    else if (layerType == "wavnetc")
    	return new WavNetCore<TDevice>(layerChild, weightsSection,
				       *precedingLayer, maxSeqLength, layerID);
    
    else if (layerType == "externalloader")
    	return new ExternalLoader<TDevice>(layerChild, weightsSection,
					   *precedingLayer, maxSeqLength, layerID);
    
    else if (layerType == "vqlayer")
    	return new vqLayer<TDevice>(layerChild, weightsSection,
				    *precedingLayer, maxSeqLength, layerID);
    
    else if (layerType == "embedding")
    	return new EmbeddingLayer<TDevice>(layerChild, weightsSection,
					   *precedingLayer, maxSeqLength, layerID);
    
    else if (layerType == "signalgen")
    	return new SignalGenLayer<TDevice>(layerChild, weightsSection,
					   *precedingLayer, maxSeqLength, layerID);
    
    else if (layerType == "filtering")
    	return new FilteringLayer<TDevice>(layerChild, weightsSection,
					   *precedingLayer, maxSeqLength, layerID);

    else if (layerType == "featextract")
    	return new FeatExtract<TDevice>(layerChild, weightsSection,
					*precedingLayer, maxSeqLength, layerID);
    else if (layerType == "gatedact")
    	return new GatedActLayer<TDevice>(layerChild, weightsSection,
					  *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "feedback_hidden")
    	return new FeedBackHiddenLayer<TDevice>(layerChild, weightsSection,
						*precedingLayer, maxSeqLength, layerID);
    else if (layerType == "sinc_filter")
    	return new SincFilterLayer<TDevice>(layerChild, weightsSection,
					    *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "self_attention")
    	return new SelfAttentionLayer<TDevice>(layerChild, weightsSection,
					       *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "mutual_weave")
    	return new InterWeaveLayer<TDevice>(layerChild, weightsSection, *precedingLayer,
					    maxSeqLength, layerID);
    else if (layerType == "layernorm_tanh")
    	return new LayerNormLayer<TDevice, Tanh>(layerChild, weightsSection,
						   *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "layernorm_logistic")
    	return new LayerNormLayer<TDevice, Logistic>(layerChild, weightsSection,
						       *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "layernorm")
    	return new LayerNormLayer<TDevice, Identity>(layerChild, weightsSection,
						       *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "layernorm_relu")
    	return new LayerNormLayer<TDevice, Relu>(layerChild, weightsSection,
						   *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "batchnorm_new_tanh")
    	return new BatchNormNewLayer<TDevice, Tanh>(layerChild, weightsSection,
						   *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "batchnorm_new_logistic")
    	return new BatchNormNewLayer<TDevice, Logistic>(layerChild, weightsSection,
						       *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "batchnorm_new")
    	return new BatchNormNewLayer<TDevice, Identity>(layerChild, weightsSection,
						       *precedingLayer, maxSeqLength, layerID);
    else if (layerType == "batchnorm_new_relu")
    	return new BatchNormNewLayer<TDevice, Relu>(layerChild, weightsSection,
						   *precedingLayer, maxSeqLength, layerID);

    else if (layerType == "simple_feedback")
    	return new SpecialFeedBackLayer<TDevice>(layerChild, weightsSection,
						 *precedingLayer, maxSeqLength, layerID);    

    else if (layerType == "sse"                       || layerType == "weightedsse"  || 
	     layerType == "rmse"                      || layerType == "ce"  || 
	     layerType == "wf"                        || layerType == "binary_classification" ||
	     layerType == "multiclass_classification" || layerType == "mdn" || 
	     layerType == "kld"                       || layerType == "dft" ||
	     layerType == "featsse"                   || layerType == "sse_multi" ||
	     layerType == "sse_cos"                   || layerType == "mutual_infor") {
        //layers::TrainableLayer<TDevice>* precedingTrainableLayer = 
	// dynamic_cast<layers::TrainableLayer<TDevice>*>(precedingLayer);
        //if (!precedingTrainableLayer)
    	//    throw std::runtime_error("Cannot add post output layer after a non trainable layer");
        if (layerType == "sse")
    	    return new SsePostOutputLayer<TDevice>(layerChild, *precedingLayer,
						   maxSeqLength, layerID);
	
        else if (layerType == "kld")
    	    return new KLPostOutputLayer<TDevice>(layerChild, *precedingLayer,
						  maxSeqLength, layerID);
	
	else if (layerType == "weightedsse")
    	    return new WeightedSsePostOutputLayer<TDevice>(layerChild,
							   *precedingLayer,
							   maxSeqLength,
							   layerID);
        else if (layerType == "rmse")
            return new RmsePostOutputLayer<TDevice>(layerChild, *precedingLayer,
						    maxSeqLength, layerID);
        else if (layerType == "ce")
            return new CePostOutputLayer<TDevice>(layerChild, *precedingLayer,
						  maxSeqLength, layerID);
	
        if (layerType == "sse_mask" || layerType == "wf") 
	    // wf provided for compat. with dev. version
    	    return new SseMaskPostOutputLayer<TDevice>(layerChild, *precedingLayer,
						       maxSeqLength, layerID);
	
        else if (layerType == "binary_classification")
    	    return new BinaryClassificationLayer<TDevice>(layerChild,
							  *precedingLayer,
							  maxSeqLength, layerID);
	
	else if (layerType == "mdn")
	    return new MDNLayer<TDevice>(layerChild, weightsSection,
					 *precedingLayer, maxSeqLength, layerID);
	else if (layerType == "dft")
	    return new DFTPostoutputLayer<TDevice>(layerChild,
						   *precedingLayer, maxSeqLength, layerID);
	else if (layerType == "featsse")
	    return new SsePostOutputLayerForFeatTrans<TDevice>(layerChild,
							       *precedingLayer, maxSeqLength,
							       layerID);
	else if (layerType == "sse_multi")
	    return new SsePostOutputMultiLayer<TDevice>(layerChild,
							*precedingLayer, maxSeqLength,
							layerID);
	else if (layerType == "sse_cos")
	    return new SseCosPostOutputLayer<TDevice>(layerChild,
						      *precedingLayer, maxSeqLength,
						      layerID);	
	else if (layerType == "mutual_infor")
	    return new MutualInforPostOutputLayer<TDevice>(layerChild,
							   *precedingLayer, maxSeqLength,
							   layerID);	
        else // if (layerType == "multiclass_classification")
    	    return new MulticlassClassificationLayer<TDevice>(layerChild,
							      *precedingLayer, maxSeqLength,
							      layerID);

    }else if (layerType == "inter_sse"){
    	return new InterMetricLayer_sse<TDevice>(layerChild, *precedingLayer,
						 maxSeqLength, layerID);
    }else if (layerType == "inter_softmax"){
    	return new InterMetricLayer_softmax<TDevice>(layerChild, *precedingLayer,
						     maxSeqLength, layerID);
    }else{
        throw std::runtime_error(std::string("Error in network.jsn: unknown type'") +
				 layerType + "'");
    }
}

template <typename TDevice>
layers::Layer<TDevice>* LayerFactory<TDevice>::createSkipNonParaLayer(
					   const std::string        &layerType,
					   const helpers::JsonValue &layerChild,
					   const helpers::JsonValue &weightsSection,
					   int                       parallelSequences, 
					   int                       maxSeqLength,
					   int                       layerID,
					   std::vector<layers::Layer<TDevice>*> &precedingLayers
					   )
{
    using namespace layers;

    if (layerType == "skipadd" || layerType == "skipini"){
	return new SkipAddLayer<TDevice>(layerChild, weightsSection,
					 precedingLayers, maxSeqLength, layerID);
    }else if (layerType == "skipcat"){
	return new SkipCatLayer<TDevice>(layerChild, weightsSection,
					 precedingLayers, maxSeqLength, layerID);
    }else if (layerType == "skipweightadd"){
	return new SkipWeightAddLayer<TDevice>(layerChild, weightsSection,
					       precedingLayers, maxSeqLength, layerID);
    }else if (layerType == "weighted_merge"){
	return new SkipMergeDim<TDevice>(layerChild, weightsSection,
					 precedingLayers, maxSeqLength, layerID);
    }else if (layerType == "normflow"){
	return new NormFlowLayer<TDevice>(layerChild, weightsSection,
					  precedingLayers, maxSeqLength, layerID);
    }else if (layerType == "structTrans"){
	return new StructTransLayer<TDevice>(layerChild, weightsSection,
					     precedingLayers, maxSeqLength, layerID);
    }else if (layerType == "distilling"){
	return new DistillingLayer<TDevice>(layerChild, weightsSection,
					    precedingLayers, maxSeqLength, layerID);
    }else if (layerType == "feattrans"){
	return new SkipForFeatTrans<TDevice>(layerChild, weightsSection,
					     precedingLayers, maxSeqLength, layerID);
    }else if (layerType == "random_shuffle"){
	return new RandomShuffleLayer<TDevice>(layerChild, weightsSection,
					       precedingLayers, maxSeqLength, layerID);
    }else{
	printf("Impossible bug\n");
	throw std::runtime_error(std::string("The layer is not skip-nonpara layer"));	
    }
}

template <typename TDevice>
layers::Layer<TDevice>* LayerFactory<TDevice>::createSkipParaLayer(
					   const std::string        &layerType,
					   const helpers::JsonValue &layerChild,
					   const helpers::JsonValue &weightsSection,
					   int                       parallelSequences, 
					   int                       maxSeqLength,
					   int                       layerID,
					   std::vector<layers::Layer<TDevice>*> &precedingLayers
					   )
{
    using namespace layers;
    using namespace activation_functions;

    if (precedingLayers.size()!=2){
	printf("Error in network.jsn: SkipParaLayer requires two input paths\n");
	printf("Please check whether skipini/skipadd is avaiable before skippara\n");
	throw std::runtime_error(std::string("Error in network.jsn"));
    }else{
	if (layerType == "skippara_tanh"){
	    return new SkipParaLayer<TDevice, Tanh>(layerChild, weightsSection,
						    precedingLayers, maxSeqLength, layerID);
	}else if(layerType == "skippara_logistic"){
	    return new SkipParaLayer<TDevice, Logistic>(layerChild, weightsSection,
							precedingLayers, maxSeqLength, layerID);
	}else if(layerType == "skippara_identity"){
	    return new SkipParaLayer<TDevice, Identity>(layerChild, weightsSection,
							precedingLayers, maxSeqLength, layerID);  
	}else if(layerType == "skippara_relu"){
	    return new SkipParaLayer<TDevice, Relu>(layerChild, weightsSection,
						    precedingLayers, maxSeqLength, layerID);
	}else{
	    printf("Type of Skippara can only be: skippara_tanh, skippara_logistic,");
	    printf("skippara_identity, skippara_relu\n");
	    throw std::runtime_error(std::string("Error in network.jsn: unknown type:")+layerType);
	}
    }
}


// explicit template instantiations
template class LayerFactory<Cpu>;
template class LayerFactory<Gpu>;
