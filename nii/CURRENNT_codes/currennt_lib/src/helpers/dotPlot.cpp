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

#include "./dotPlot.hpp"

namespace dotPlot{

    void printDotHead(std::ofstream& ofs)
    {
	ofs << "digraph G{\n";
    }

    void printDotNode(std::ofstream& ofs,
		      const int src_id, const std::string src_name, const std::string src_type,
		      const int tar_id, const std::string tar_name, const std::string tar_type)
    {
	
	ofs << "\"" << src_id << "_" << src_type << "\\n" << src_name << "\"" << " -> ";
	ofs << "\"" << tar_id << "_" << tar_type << "\\n" << tar_name << "\"" << ";\n" << std::endl;
	//ofs << "\"" << src_id << "_" << src << "\"" << " -> ";
	//ofs << "\"" << tar_id << "_" << tar << "\"" << ";\n" << std::endl;

    }
    
    void printDotNode(std::ofstream& ofs, const std::string src, const std::string tar)
    {
	ofs << src << " -> " << tar << ";\n" << std::endl;
    }

    void printDotEnd(std::ofstream& ofs)
    {
	ofs << "}" << std::endl;;
    }

}
