#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <iostream>
#include "Utilities.hpp"

class Network
{
private:
	double eta_;	// learning rate
	std::array<Matrice, _NB_LAYERS_ - 1> weights_;
	MatriceFixe neurons_;
	unsigned int iterations_tot_;
	Matrice correctOutputs_;
	MatriceFixe deltas_;		
	std::vector<AllInfos> wholeData_;
	double delta_final_scal_;
	int data_size_;
	Matrice bias_;
	
public:
	//---------Utilities--------------
	MatriceFixe getActivations() const;
	Matrice getCorrectOutput() const;
	void calculate_delta_final_scal();
	std::vector<double> reArrangeVect(int indexLayer, int indexTargetNeur) const;
	void displayLoadingBar(int i) const;
	void writeWeights(std::vector<std::ofstream>& out, int step) const;
	void writeSingleWeight(std::ostream& out, int layer, int step) const;
	
	void buildRandomWeights();
	void buildRandomBiases();
	void readWholeInput(std::ifstream& inputFile);
	Layer computeError(std::ifstream& fichier) const;
	
	void run();
	void update(std::ofstream& errorsFile, std::vector<std::ofstream>& weightFiles, int step, std::vector<int>& deja_tires);
	void activateLayer(int index);
	
	//-----BackPropagation-----------
	void calculateDeltas(int randomStep);
	void deltaLayer(int index);
	void updateWeights();
	void updateBiases();


	Network(unsigned int iterations_tot = 100, double learningRate = 0.5);
	~Network();
};

#endif



