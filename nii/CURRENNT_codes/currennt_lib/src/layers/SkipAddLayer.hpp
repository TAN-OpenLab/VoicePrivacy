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
/****
 *
 *
 *
 ****/

#ifndef LAYERS_SKIPADDLAYER_HPP
#define LAYERS_SKIPADDLAYER_HPP

#include "SkipLayer.hpp"
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace layers {
    
    /**********************************************************************
	    Definition of the Skip layer
     A base class for SkipAdd and SkipPara layers

     **********************************************************************/
    
    // class definition
    template <typename TDevice>
    class SkipAddLayer : public SkipLayer<TDevice>
    {
	typedef typename TDevice::real_vector    real_vector;
        
    private:

	// this layer SkipInit or SkipAdd
	bool                         m_flagSkipInit; 
	bool                         m_virtualLayer;
	real_t                       m_noiseRatio;
	std::string                  m_previousSkipStr;
    public:
		
	// Construct the layer
	SkipAddLayer(
		     const helpers::JsonValue &layerChild,
		     const helpers::JsonValue &weightsSection,
		     std::vector<Layer<TDevice>*> &precedingLayers,
		     int maxSeqLength,
		     int layerID
		     );

	// Destructor
	virtual ~SkipAddLayer();
	
	// void 
	virtual const std::string& type() const;

	// NN forward
	virtual void computeForwardPass(const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// fake output from gate
	real_vector& outputFromGate();

	// return all the preceding layers
	//std::vector<Layer<TDevice>*> PreLayers();

	// NN forward
	virtual void computeForwardPass(const int timeStep, const int nnState);

	// NN backward
	virtual void computeBackwardPass(const int timeStep, const int nnState);

	// return reference to the m_outputErrorsFromSkipLayer
	// real_vector& outputErrorsFromSkipLayer();

	// output the layer configuration
	virtual void exportLayer(const helpers::JsonValue &layersArray,
				 const helpers::JsonAllocator &allocator) const;

	virtual real_vector& outputs();

	virtual void reduceOutputBuffer();

	virtual int outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);

	void logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng, bool flag_add);
	
	void swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng, bool flag_get);	

    };

}


#endif 


