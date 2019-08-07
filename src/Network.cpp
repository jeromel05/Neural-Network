#include "Network.hpp"

bool Network::checkActivationBounds(double valeur) const
{
	if(valeur < 0 or valeur > 1.0){
		 return false;
	}else{
		return true;
	}
}

double Network::flowerTypeToDouble(const std::string& type1) const
{
	if(type1.compare(std::string("Iris-setosa")) ==0){
		return 0.0;
	}else if(type1.compare(std::string("Iris-versicolor")) ==0){
		return 1.0;
	}else if(type1.compare(std::string("Iris-virginica")) ==0){
		return 2.0;
	}else{
		throw(std::string("undefined flower type"));
		return -10;
	}
}


Layer Network::prodElement(const Layer& tab1, const Layer& tab2) const
{
	assert(tab1.size() == tab2.size());
	Layer res;
	for(size_t i(0); i < tab1.size(); ++i){
		res.push_back(tab1[i] * tab2[i]);
	}
return res;
}

double Network::prodScal(const Layer& tab1, const Layer& tab2) const
{
	assert(tab1.size() == tab2.size());
	double res(0.0);
	for(size_t i(0); i < tab1.size(); ++i){
		res += tab1[i] * tab2[i];
	}
return res;
}

double  Network::sum(const Layer& tab1) const
{
	double res(0.0);
	for(size_t i(0); i < tab1.size(); ++i){
		res+=tab1[i];
	}
	return res;
}

Layer  Network::vectSum(const Layer& tab1,const Layer& tab2) const
{
	assert(tab1.size() == tab2.size());
	Layer res;
	for(size_t i(0); i < tab1.size(); ++i){
		res.push_back(tab1[i] + tab2[i]);			//devrait tout convertir en array pour vitesse, size n'est jamais mod dans programme
	}
	return res;
}

Layer  Network::vectSub(const Layer& tab1,const Layer& tab2) const
{
	assert(tab1.size() == tab2.size());
	Layer res;
	for(size_t i(0); i < tab1.size(); ++i){
		res.push_back(tab1[i] - tab2[i]);			//devrait tout convertir en array pour vitesse, size n'est jamais mod dans programme
	}
	return res;
}



void   Network::afficheVect(const Layer& tab) const
{
	for(auto i: tab){
		std::cout << i << " ";
	}
	std::cout << std::endl;
}

double   Network::sigmoid(double valeur) const
{ 
	if(valeur > 1e+10){
		std::cerr << "val:" << valeur << std::endl;
		valeur = 1e+10;
		throw std::string("out of bounds activation");
	}
	double temp(0.0);
	temp = 1 / (1 + exp(-valeur));
	assert(temp <= 1.0 and temp >= 0.0);
	return temp;
}

double  Network::deriveeSigmoid(double valeur) const
{
	return sigmoid(valeur) * (1 - sigmoid(valeur));
}

int  Network::newInt(std::vector<int>& deja_tires) const
{
	if(deja_tires.size() >= data_size_) deja_tires.clear();
	std::uniform_int_distribution<> dint(0,data_size_-1);	//on veut tirer au hasard dans les données pour ne pas tirer les données dans l'ordre
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

MatriceFixe Network::getActivations() const
{
	return neurons_;
}

Matrice Network::getCorrectOutput() const
{
	return correctOutputs_;
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
	std::uniform_real_distribution<double> d(-0.4,0.4);
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

void Network::buildRandomBiases()
{
	std::uniform_real_distribution<double> d(-0.1,0.1);
	Layer temp;
	
	for(size_t h(0); h < _NB_LAYERS_ - 1; ++h){
		for(size_t i(0); i < neurons_[h + 1].size(); ++i){
			temp.push_back(d(gen));
		}
		bias_.push_back(temp);
		temp.clear();
	}
}

//check taille fichier input >= nb iterations
//iniitiliser correctOutputs a un vec de zeros
void Network::readWholeInput(std::ifstream& inputFile)
{	
	while(!inputFile.eof()){
		Layer mesures;
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
		temp3 = flowerTypeToDouble(temp2);
		Layer temp_vec(3,0.0);		//push back un vect avec un "1" à la bonne position
		temp_vec[temp3] += 1;
		correctOutputs_.push_back(temp_vec);
	
		wholeData_.push_back(std::make_pair(mesures,temp3));
		mesures.clear();
	}
	data_size_ = wholeData_.size();
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
		update(errorsFile, weightFiles, h, deja_tires);
	}
	std::cout << std::endl;
	
	out1.close();
	out2.close();
	errorsFile.close();
	inputFile.close();
	std::cout << "Plotting graphs..." << std::endl;
}

void Network::update(std::ofstream& errorsFile, std::vector<std::ofstream>& weightFiles, int step, std::vector<int>& deja_tires)
{	
	int randomStep = newInt(deja_tires);	//newInt nous donne une valeur que l'on a pas encore tiré entre 0 et 149

	neurons_[0] = wholeData_[randomStep].first;
	
	for(size_t i(0); i < neurons_.size() - 1; ++i){
		activateLayer(i);
	}
		
	calculateDeltas(randomStep);
	updateWeights();
	updateBiases();
	writeWeights(weightFiles, step);
	displayLoadingBar(step + 1);			//+1 pour arriver à 100%
	
	if(step % (iterations_tot_/300) == 0){													//on prend l'erreur prop au nb d'iteration tot --> allège le graph
		errorsFile << delta_final_scal_ << '\n';
	}
	
	std::cout << "out corr: ";   afficheVect(correctOutputs_[randomStep]);
	  afficheVect(neurons_[_NB_LAYERS_-1]);
	std::cout << std::endl;
	
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
    std::cout << "] " << int(progress * 100) << "%\r";
	std::cout.flush();     
}

void Network::calculateDeltas(int randomStep)
{
	deltas_[_NB_LAYERS_ - 1] =   vectSub(correctOutputs_[randomStep],neurons_[_NB_LAYERS_ - 1]); //calcule le dernier layer
	calculate_delta_final_scal();		//least squares scalar value
	for(size_t i(_NB_LAYERS_ - 1); i > 1 ; --i){	//layer 0 is input layer so no need to compute deltas
		deltaLayer(i);
	}
}

void Network::calculate_delta_final_scal(){
	delta_final_scal_ = sqrt(prodScal(deltas_[_NB_LAYERS_-1],deltas_[_NB_LAYERS_-1]))/_NB_OUTPUTS_;
	//std::cout <<"d: " << delta_final_scal_ << std::endl;
}

void Network::deltaLayer(int index)
{
	Layer temp_tab;
	Layer tab1;
	
	tab1 = deltas_[index];
	
	for(size_t i(0); i < neurons_[index - 1].size(); ++i){	
		for(size_t j(0); j < neurons_[index].size(); ++j){	
			temp_tab.push_back(weights_[index - 1][j][i]);
		}
		deltas_[index - 1][i] =   prodScal(tab1, temp_tab);
		temp_tab.clear();
	}
}	

void Network::activateLayer(int index)
{
	for(size_t i(0); i < neurons_[index + 1].size(); ++i){	
		neurons_[index + 1][i] =   sigmoid(prodScal(neurons_[index],  weights_[index][i]) + bias_[index][i]);	//SUPER GROSSE ERREUR -> BIAS DANS SIGMOID!!!
	}
}

//aout : conclusion: fonction d'update paraissent juste mais il y a forcement une fate pck weights become way too big -> rearranger tout sosu forme matricielle en définissant le produit matriciel et Hadamard 

void Network::updateWeights()			//cette fonction cause PAS segfault, mais value of activation of neurons shoots up to 1e+300			
{
	for(size_t i(0); i < weights_.size(); ++i){
		for(size_t j(0); j < weights_[i].size(); ++j){
			for(size_t k(0); k < weights_[i][j].size(); ++k){
				weights_[i][j][k] += eta_ *   sum(prodElement(deltas_[i+1],reArrangeVect(i,k)))/deltas_[i+1].size() *   deriveeSigmoid(neurons_[i][k]) * neurons_[i][k];
				if( weights_[i][j][k] < -100 or weights_[i][j][k] > 100){
					std::cerr << "w: " << weights_[i][j][k] << std::endl;
				}
				//weights_[i][j][k] += eta_ * deltas_[i + 1][j] * deriveeSigmoid(neurons_[i + 1][j]) * neurons_[i][k]; // + 1  car 1er layer est input -> pas de delta, + car le moins s'annule avec celui de la dérivée partielle
				//std::cerr << "une fois" << neurons_[i][k] << std::endl;
				//Do the formula w/ sum pck multiple outputs requires it
			}
		}
	}
}

void Network::updateBiases()
{
	for(size_t i(0); i < _NB_LAYERS_-1; ++i){
		for(size_t j(0); j < neurons_[i + 1].size(); ++j){
			//bias_[0][j] += eta_ * deltas_[1][j] * deriveeSigmoid(neurons_[1][j]);	//version for scalar output -> works correctly!!
			bias_[i][j] += eta_ *   sum(  prodElement(deltas_[i+1],reArrangeVect(i,j)))/deltas_[i+1].size() *   deriveeSigmoid(neurons_[1][j]);	//dernière version, 80% confident que correct avant aout
			//bias_[h][j] += eta_ * sum(prodElement(deltas_[h+1],weights_[h+1][j]))/deltas_[h+1].size() * deriveeSigmoid(neurons_[1][j]);//pb pck maps connections the wrong way (avant rearrangement)
		}
	}
}

Layer Network::reArrangeVect(int indexLayer, int indexTargetNeur) const
{
	Layer res;
	for(size_t j(0); j < neurons_[indexLayer + 1].size(); ++j){
		res.push_back(weights_[indexLayer][j][indexTargetNeur]);
	}
	return res;
}	

Network::Network(unsigned int iterations_tot, double learningRate)
:eta_(learningRate), iterations_tot_(iterations_tot), delta_final_scal_(0.0), data_size_(0)
{
	
	for(size_t i(0); i < _NB_INPUTS_; ++i){
		neurons_[0].push_back(0.0);
	}	
	for(size_t i(0); i < _NB_NEURONS1_; ++i){
		neurons_[1].push_back(0.0);
	}
	for(size_t i(0); i < _NB_OUTPUTS_; ++i){
		neurons_[2].push_back(0.0);
	}	
	buildRandomWeights();
	buildRandomBiases();
	deltas_ = neurons_;
}

Network::~Network()
{}


