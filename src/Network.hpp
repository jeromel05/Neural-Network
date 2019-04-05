#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <string>
#include <random>
#include <sstream>
#include <array>
#include <stdio.h>
#include <unistd.h>

#define _NB_LAYERS_ 3 													//counting input and output layer
#define _NB_INPUTS_ 4
#define _NB_NEURONS1_ 1000
#define _NB_OUTPUTS_ 1

typedef double Neuron;
typedef std::vector<double> Layer;
typedef std::vector<std::vector<double> > Matrice;
typedef std::array<std::vector<double>, _NB_LAYERS_> MatriceFixe;
typedef std::pair<std::vector<double>, double> AllInfos;

static std::random_device rd;
static std::mt19937 gen(rd());

class Network
{
private:
	double eta_;	// learning rate
	std::array<Matrice, _NB_LAYERS_ - 1> weights_;
	MatriceFixe neurons_;
	unsigned int iterations_tot_;
	std::vector<int> correctOutputs_;
	MatriceFixe deltas_;		
	double delta_final_;
	std::vector<AllInfos> wholeData_;
	
public:
	//---------Utilities--------------
	MatriceFixe getActivations() const;
	std::vector<int> getCorrectOutput() const;
	double sigmoid(double valeur) const;
	double deriveeSigmoid(double valeur) const;
	bool checkActivationBounds(double valeur) const;
	int classToInt(char cla) const;
	char intToClass(int i) const;
	double flowerTypeToDouble(std::string type1) const;
	int newInt(std::vector<int>& deja_tires) const;

	
	std::vector<double> prodElement(const std::vector<double>& tab1, const std::vector<double>& tab2) const;
	double prodScal(const std::vector<double>& tab1, const std::vector<double>& tab2) const;
	void afficheVect(const Layer& tab) const;
	void afficheMatrice(const Matrice& mat) const;
	void afficheMatrice(const MatriceFixe& mat) const;
	void displayLoadingBar(int i) const;
	void afficheWeights() const;
	void writeWeights(std::vector<std::ofstream>& out, int step) const;
	void writeSingleWeight(std::ostream& out, int layer, int step) const;
	
	void generateTrainingDataSet() const;
	void buildRandomWeights();
	std::vector<double> readInput(std::ifstream& inputFile);
	void readWholeInput(std::ifstream& inputFile);
	std::vector<double> computeError(std::ifstream& fichier) const;
	
	void run();
	void update(std::ofstream& errorsFile, std::vector<std::ofstream>& weightFiles, int step, std::vector<int>& deja_tires);
	void activateLayer(int index);
	
	//-----BackPropagation-----------
	void calculateDeltas();
	void deltaLayer(int index);
	void updateWeights();

	Network(unsigned int iterations_tot = 100, double learningRate = 0.5);
	~Network();
};

#endif



