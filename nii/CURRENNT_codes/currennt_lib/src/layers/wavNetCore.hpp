/******************************************************************************
 * This file is an addtional component of CURRENNT. 
 * Xin WANG
 * National Institute of Informatics, Japan
 * 2016
 *
 * This file is part of CURRENNT. 
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
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
/*
 */

#ifndef LAYERS_WAVNETCORE_HPP
#define LAYERS_WAVNETCORE_HPP


#include "TrainableLayer.hpp"

namespace layers{

    template <typename TDevice>
    class WavNetCore : public TrainableLayer<TDevice>
    {
	typedef typename TDevice::real_vector     real_vector;
	typedef typename Cpu::real_vector         cpu_real_vector;
	typedef typename TDevice::int_vector      int_vector;
	typedef typename TDevice::pattype_vector  pattype_vector;
	typedef typename Cpu::int_vector          cpu_int_vector;

    private:
	int            m_wavCoreOpt;     // option
	int            m_contextDim;     // dimension of the textual feature

	real_vector    m_coreBuf;        // internal data buffer
	real_vector    m_contextBuf;     // buffer for the textual data
	real_vector    m_contextGraBuf;
	real_vector    m_contextTanhBuf; 
	real_vector    m_contextRawBuf;
	real_vector    m_contextMV;
	std::string    m_contextMVStr;
	int            m_contextCurMaxLength;
	
	TrainableLayer<TDevice> *m_exInputLayer;
	
	bool                     m_iniWavCoreC;
	bool                     m_exInputContextResone;
	WavNetCore<TDevice>     *m_iniWavCPtr;

	void __loadContextBuff();
	void __allocateLocalMem();
	void __clearLocalMem();
	
    public:
	WavNetCore(
	    const helpers::JsonValue &layerChild,
	    const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer,
	    int                       maxSeqLength,
	    int                       layerID
	);

	virtual ~WavNetCore();
	
	
	virtual const std::string& type() const;
	
	// NN forward
	virtual void computeForwardPass(const int nnState);
	
	// NN forward, per frame
	virtual void computeForwardPass(const int timeStep, const int nnState);
	
	// NN backward
	virtual void computeBackwardPass(const int nnState);

	// NN backward
	virtual void computeBackwardPass(const int timeStep, const int nnState);

	// export
	virtual void exportLayer(const helpers::JsonValue &layersArray, 
				 const helpers::JsonAllocator &allocator) const;
	//
	virtual void loadSequences(const data_sets::DataSetFraction &fraction, const int nnState);

	// load the target data from the target layer
	void linkTargetLayer(Layer<TDevice> &targetLayer);

	const real_vector& context() const;

	virtual void reduceOutputBuffer();
	
	virtual int  outputBufPtrBias(const int timeStepTimesParallel, const int nnState);

	std::vector<int> dependLayerIDs();

	void clearAllBuffers();

	void resizeAllBuffers(const int timeLength);

	void logAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng, bool flag_add);
	
	void swapAllBuffers(helpers::vecPoolManager<TDevice> &vecPoolMng, bool flag_get);

	
    };
    
}

#endif
