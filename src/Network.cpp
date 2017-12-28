#include "Network.hpp"

Matrice Network::getActivations() const
{
	return neurons_;
}

std::vector<int> Network::getCorrectOutput() const
{
	return correctOutputs_;
}

double Network::sigmoid(double valeur) const
{
	double temp(0.0);
	temp = 1 / (1 + exp(-valeur));
	assert(temp <= 1.0 and temp >= 0.0);
	return temp;
}

double Network::deriveeSigmoid(double valeur) const
{
	return sigmoid(valeur) * (1 - sigmoid(valeur));
}

bool Network::checkActivationBounds(double valeur) const
{
	if(valeur < 0 or valeur > 1.0){
		 return false;
	}else{
		return true;
	}
}

std::vector<double> Network::prodElement(const std::vector<double>& tab1, const std::vector<double>& tab2) const
{
	assert(tab1.size() == tab2.size());
	std::vector<double> res;
	for(size_t i(0); i < tab1.size(); ++i){
		res.push_back(tab1[i] * tab2[i]);
	}
return res;
}

double Network::prodScal(const std::vector<double>& tab1, const std::vector<double>& tab2) const
{
	assert(tab1.size() == tab2.size());
	double res(0.0);
	for(size_t i(0); i < tab1.size(); ++i){
		res += tab1[i] * tab2[i];
	}
return res;
}

void Network::afficheMatrice(const Matrice& mat) const
{
	for(auto layer: mat){
		afficheVect(layer);
	}
}

void Network::afficheVect(const Layer& tab) const
{
	for(auto i: tab){
		std::cout << i << " ";
	}
	std::cout << std::endl;
}

void Network::generateTrainingDataSet() const
{
	std::ofstream out;
	out.open("../data/trainingDataSet.txt");

	double total(0.0);
	double temp(0.0);
	std::uniform_int_distribution<> dist(-10, 10);
	
	for(size_t i(0); i < iterations_tot_; ++i){
		for(size_t j(0); j < 2; ++j){
			temp = dist(gen);
			total += temp;
			out << temp << " ";
		}
		out << "=";
			if(total >= 10){
				out << 1;
			}else{
				out << 0;
			}
		out << '\n';
		total = 0.0;
	}
	out.close();
}

void Network::buildRandomWeights()
{
	std::uniform_real_distribution<double> d(-1.0,1.0);
	Matrice temp1;
	Layer temp2;
	
	for(size_t h(0); h < _NB_LAYERS_ - 1; ++h){
		for(size_t i(0); i < neurons_[h + 1].size(); ++i){
			for(size_t j(0); j < neurons_[h].size(); ++j){
					temp2.push_back(d(gen));
				}
			temp1.push_back(temp2);
			temp2.clear();
			}
		
		weights_[h] = temp1;
		temp1.clear();
	}
	
	for(auto a: weights_){
		afficheMatrice(a);
		std::cout << "weights sheet" << std::endl;
	}
		
}

/*
std::vector<double> Network::computeError(std::ifstream& fichier) const
{
	std::vector<double> res;
	std::string line;

	fichier.ignore(100, '=');
		for(size_t i(0); i < 3; ++i){

			getline(fichier, line, ',');
			res.push_back(stod(line));
			if(res[i] < 0){
				throw error + std::string(_NEGATIVE_ARGUMENT_);
			}
			line.clear();
		}
	return res;
}

	if(!checkActivationBounds(res)){
		throw "Correct output at position " << i << " is out of bounds";
	}
	return res;
}
*/

std::vector<double> Network::readInput(std::ifstream& inputFile)
{
	std::vector<double> res;
	std::string line;

		getline(inputFile, line);
		std::istringstream streamDouble(line);
			
		double temp1(0.0);
			while(streamDouble >> temp1){
				res.push_back(temp1);
			}

		char temp2 = line.back();
		int temp3 = temp2 - '0';			//(char)'0' = (int)48 en ASCII --> à soustraire
		assert(temp3 >= 0 and temp3 <= 9);
		correctOutputs_.push_back(temp3);	
	return res;
}

void Network::update()
{
	std::string filePath("../data/trainingDataSet.txt");
	std::ifstream inputFile;
	inputFile.open(filePath);
		if(inputFile.fail()) {
			throw "Failed to open inputFile";
		}
		
	for(size_t h(0); h < iterations_tot_; ++h){
		std::vector<double> inputs(readInput(inputFile));
		std::cout << "Iteration " << h << std::endl;
			/*
				std::cout << " in: " << '\t';
					for(auto i: inputs){
						std::cout << i << '\t';
					}
					std::cout << "out: " << '\t' << correctOutputs_[h];
					std::cout << std::endl;
				
			*/
		for(size_t i(0); i < neurons_.size() - 1; ++i){
			activateLayer(i, inputs);
		}
		afficheMatrice(neurons_);
		//computeOutput(); 
	}
	inputFile.close();
}

void Network::activateLayer(int index, const std::vector<double>& inputs)
{
	std::vector<double> tab1;
	if(index <= 0){														
		tab1 = inputs;													//first layer gets activated by inputs
	}else{
		tab1 = neurons_[index];										//other layers are activated by previous layer
	}
	
	for(size_t i(0); i < neurons_[index + 1].size(); ++i){	
		//std::cerr << index << tab1.size() << weights_[index][i].size() << std::endl;
		neurons_[index + 1][i] = sigmoid(prodScal(tab1,  weights_[index][i]));
	}
}
	

Network::Network(unsigned int iterations_tot, double learningRate)
:learningRate_(learningRate), iterations_tot_(iterations_tot)
{
	neurons_.push_back({0.0, 0.0});
	neurons_.push_back({0.0, 0.0, 0.0});
	neurons_.push_back({0.0, 0.0});
	neurons_.push_back({0.0});
	buildRandomWeights();
}

Network::~Network()
{}


