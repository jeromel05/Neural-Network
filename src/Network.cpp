#include "Network.hpp"

MatriceFixe Network::getActivations() const
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

double Network::flowerTypeToDouble(std::string type1) const
{
	if(type1.compare(std::string("Iris-setosa")) ==0){
		return 0.0;
	}else if(type1.compare(std::string("Iris-versicolor")) ==0){
		return 1.0;
	}else{
		throw(std::string("undefined flower type"));
		return -10;
	}
}

int Network::classToInt(char cla) const	//serait utile pour balance data
{
	switch(cla){
		case 'L':
			return -1;
		case 'B':
			return 0;
		case 'R':
			return 1;
		default:
			return 0;
	}	
}

char Network::intToClass(int i) const
{
	switch(i){
		case -1:
			return 'L';
		case 0:
			return 'B';
		case 1:
			return 'R';
		default:
			return 'B';
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

void Network::afficheWeights() const
{
	for(size_t i(0); i < weights_.size(); ++i){
		for(size_t j(0); j < weights_[i].size(); ++j){
			for(size_t k(0); k < weights_[i][j].size(); ++k){
				std::cout << weights_[i][j][k] << '\t';
			}
			std::cout << std::endl;
		}
		std::cout << "nouvelle dimension" << std::endl;
	}				
}


void Network::afficheMatrice(const Matrice& mat) const
{
	for(auto layer: mat){
		afficheVect(layer);
	}
}

void Network::afficheMatrice(const MatriceFixe& mat) const
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
	std::uniform_int_distribution<> dist(0, 1);
	
	for(size_t i(0); i < iterations_tot_; ++i){
		for(size_t j(0); j < 2; ++j){
			temp = dist(gen);
			total += temp;
			out << temp << " ";
		}
		out << "=";
			if(total >= 2){
				out << 1;
			}else{
				out << 0;
			}
		out << '\n';
		total = 0.0;
	}
	out.close();
}

void Network::writeWeights(std::vector<std::ofstream>& out, int step) const
{	
	for(size_t i(0); i < weights_.size(); ++i){
		writeSingleWeight(out[i], i, step);
	}
}

void Network::writeSingleWeight(std::ostream& out, int layer, int step) const
{
	// | entre les valeurs, * entre les lignes, \n entre les époques
	assert(layer >= 0 and layer < weights_.size() and step >= 0);
	
	for(size_t j(0); j < weights_[layer].size(); ++j){
			for(size_t k(0); k < weights_[layer][j].size(); ++k){
				if(k < weights_[layer][j].size() - 1){
					 out << weights_[layer][j][k] << '|';
				}else{
					out << weights_[layer][j][k];
				}
			}
			if(j < weights_[layer].size() - 1){
				out << '|';			//temporairement un | mais normalement on mettra un * entre les connections arrivant à chaque neurone séparé
			}
		}
	out << '\n';
}	

void Network::buildRandomWeights()
{
	std::uniform_real_distribution<double> d(-0.3,0.3);
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
}

std::vector<double> Network::readInput(std::ifstream& inputFile)
{
	std::vector<double> res;
	//stringstream est un buffer qui contient les données (string), pour ensuite les transformer en double
	std::string cell("");
	double temp1(0.0);
	
	for(size_t i(0); i < _NB_INPUTS_;++i){
		std::getline(inputFile,cell, ',');	//lit une seule donnée (double)
		std::stringstream(cell)>>temp1;
		res.push_back(temp1);
	}
	
	std::string temp2("");
	inputFile >> temp2;		//parsing the last information, the name of the flower
	
	double temp3(0.0);
	temp3 = flowerTypeToDouble(temp2);
	correctOutputs_.push_back(temp3);
	std::cout << temp2 << " " << temp3 << std::endl;	
	return res; //res est un vesteur de double avec les 4 mesures de la fleur
}

void Network::readWholeInput(std::ifstream& inputFile)
{	
	while(!inputFile.eof()){
		std::vector<double> mesures;
		std::string cell("");
		double temp1(0.0);
		
		for(size_t i(0); i < _NB_INPUTS_;++i){
			std::getline(inputFile,cell, ',');	//lit une seule donnée (double)
			std::stringstream(cell)>>temp1;	//stringstream est un buffer qui contient les données (string), pour ensuite les transformer en double
			mesures.push_back(temp1);
			//std::cerr << "t " << temp1;
		}
		//std::cerr << std::endl;
		
		std::string temp2("");
		inputFile >> temp2;		//parsing the last information, the name of the flower
		
		double temp3(0.0);
		//std::cerr << "t2 " << temp2 << std::endl;
		temp3 = flowerTypeToDouble(temp2);
		correctOutputs_.push_back(temp3);
		wholeData_.push_back(std::make_pair(mesures,temp3));
		mesures.clear();
	}
}

void Network::run()
{
	std::string filePath("../data/iris.txt");
	std::ifstream inputFile;
	inputFile.open(filePath);
		if(inputFile.fail()) {
			throw "Failed to open inputFile";
		}
		
	std::ofstream errorsFile;
	errorsFile.open("../data/final_errors.txt");
	
	std::ofstream out1;
	out1.open("../data/weights1.txt");
	std::ofstream out2;
	out2.open("../data/weights2.txt");
	
	std::vector<std::ofstream> weightFiles;
	weightFiles.push_back(std::move(out1));								// Impossible de push back un std::ofstream car pas de copie permise
	weightFiles.push_back(std::move(out2));
	
	std::vector<int> deja_tires;
	readWholeInput(inputFile);
	
	std::cout << "Simulation is running..." << std::endl;
		
	for(size_t h(0); h < iterations_tot_; ++h){
		//update(inputFile, errorsFile, weightFiles, h);
		update(errorsFile, weightFiles, h, deja_tires);
	}
	std::cout << std::endl;
	
	out1.close();
	out2.close();
	errorsFile.close();
	inputFile.close();
}

void Network::update(std::ofstream& errorsFile, std::vector<std::ofstream>& weightFiles, int step, std::vector<int>& deja_tires)
{	
	int randomStep(0);
	std::uniform_int_distribution<> dint1(0,100-1);
	//randomStep = dint1(gen);		//on prend plusieurs fois la meme donnée
	randomStep = newInt(deja_tires);	//newInt nous donne une valeur que l'on a pas encore tiré entre 0 et 99

	for(size_t j(0); j < _NB_INPUTS_; ++j){
		neurons_[0][j] = wholeData_[randomStep].first[j];
	}
	
	for(size_t i(0); i < neurons_.size() - 1; ++i){
		activateLayer(i);
	}
	delta_final_ = correctOutputs_[randomStep] - neurons_[_NB_LAYERS_ - 1][0];
	std::cout << " out: " << neurons_[_NB_LAYERS_ - 1][0] << " correct: " << correctOutputs_[randomStep] << " step: " << randomStep << std::endl;
	if(step % 1 == 0){													//on prend l'erreur chaque 10 itérations --> allège le graph
		errorsFile << delta_final_ << '\n';
	}
		
	calculateDeltas();
	updateWeights();
	writeWeights(weightFiles, step);
	//displayLoadingBar(step + 1);			//+1 pour arriver à 100%
}

void Network::displayLoadingBar(int i) const
{
	double progress(i / (double)iterations_tot_);
    int barWidth(70);

    std::cout << "[";
    int pos = barWidth * progress;
    for (size_t i (0); i < barWidth; ++i) {
        if (i <= pos) std::cout << "=";
        else std::cout << " ";
    }
    std::cout << "] " << progress * 100.0 << "%\r";
	std::cout.flush();     
}

void Network::calculateDeltas()
{
	for(size_t i(_NB_LAYERS_ - 1); i > 1 ; --i){	//layer 0 is input layer so no need to compute deltas
		deltaLayer(i);
	}
}

void Network::deltaLayer(int index)
{
	Layer temp_tab;
	Layer tab1;
	
	if(index >= _NB_LAYERS_ - 1){														
		tab1.push_back(delta_final_);
		deltas_[index][0] = delta_final_;											
	}else{
		tab1 = deltas_[index];						
	}
	
	for(size_t i(0); i < neurons_[index - 1].size(); ++i){	
		for(size_t j(0); j < neurons_[index].size(); ++j){	
			temp_tab.push_back(weights_[index - 1][j][i]);
		}
		deltas_[index - 1][i] = prodScal(tab1, temp_tab);
		//std::cerr << "tailles " << tab1.size() << " part " << temp_tab.size() << std::endl;
		temp_tab.clear();
	}
}	

void Network::activateLayer(int index)
{
	for(size_t i(0); i < neurons_[index + 1].size(); ++i){	
		//std::cerr << index << tab1.size() << weights_[index][i].size() << std::endl;
		neurons_[index + 1][i] = sigmoid(prodScal(neurons_[index],  weights_[index][i]));
	}
}

void Network::updateWeights()								// !!!!!!!!!!!!Problème dans la fonction: weights de input à L1 ne sont pas updaté
{
	for(size_t i(0); i < weights_.size(); ++i){
		for(size_t j(0); j < weights_[i].size(); ++j){
			for(size_t k(0); k < weights_[i][j].size(); ++k){
				weights_[i][j][k] += eta_ * deltas_[i + 1][j] * deriveeSigmoid(neurons_[i + 1][j]) * neurons_[i][k]; // + 1  car 1er layer est input -> pas de delta
				//std::cerr << "une fois" << neurons_[i][k] << std::endl;
			}
		}
	}
}

int Network::newInt(std::vector<int>& deja_tires) const
{
	std::uniform_int_distribution<> dint(0,100-1);	//on veut tirer au hasard dans les données pour ne pas tirer les données dans l'ordre
	double index1(0.0);
	bool isNew(false);
	
	while(!isNew){
		index1 = dint(gen);
		isNew = true;
		for(size_t i(0); i < deja_tires.size(); ++i){
			if(deja_tires[i] == index1) isNew = false;
		}
		deja_tires.push_back(index1);
	}
	std::cout << "randstep: " << index1 << std::endl;
	return index1;
}
	

Network::Network(unsigned int iterations_tot, double learningRate)
:eta_(learningRate), iterations_tot_(iterations_tot)
{
	for(size_t i(0); i < _NB_LAYERS_; ++i){
		neurons_[0].push_back(0.0);
	}	
	for(size_t i(0); i < _NB_NEURONS1_; ++i){
		neurons_[1].push_back(0.0);
	}
	neurons_[2] = {0.0};
	
	buildRandomWeights();
	deltas_ = neurons_;
}

Network::~Network()
{}


