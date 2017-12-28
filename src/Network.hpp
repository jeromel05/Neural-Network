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

#define _NB_LAYERS_ 4 													//counting input and output layer
#define _NB_INPUTS_ 2
#define _NB_NEURONS1_ 3
#define _NB_NEURONS2_ 2
#define _NB_OUTPUTS_ 1

typedef double Neuron;
typedef std::vector<double> Layer;
typedef std::vector<std::vector<double> > Matrice;

static std::random_device rd;
static std::mt19937 gen(rd());

class Network
{
private:
	double learningRate_;
	std::array<Matrice, _NB_LAYERS_ - 1> weights_;
	Matrice neurons_;
	unsigned int iterations_tot_;
	std::vector<int> correctOutputs_;
	
public:
	Matrice getActivations() const;
	std::vector<int> getCorrectOutput() const;
	double sigmoid(double valeur) const;
	double deriveeSigmoid(double valeur) const;
	bool checkActivationBounds(double valeur) const;
	
	std::vector<double> prodElement(const std::vector<double>& tab1, const std::vector<double>& tab2) const;
	double prodScal(const std::vector<double>& tab1, const std::vector<double>& tab2) const;
	void afficheVect(const Layer& tab) const;
	void afficheMatrice(const Matrice& mat) const;
	
	void generateTrainingDataSet() const;
	void buildRandomWeights();
	std::vector<double> readInput(std::ifstream& inputFile);
	std::vector<double> computeError(std::ifstream& fichier) const;
	
	void update();
	void activateLayer(int index, const std::vector<double>& inputs);

	Network(unsigned int iterations_tot = 10, double learningRate = 0.1);
	~Network();
};

#endif



