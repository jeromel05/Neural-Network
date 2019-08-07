#include "Utilities.hpp"

bool Utilities::checkActivationBounds(double valeur)
{
	if(valeur < 0 or valeur > 1.0){
		 return false;
	}else{
		return true;
	}
}

double Utilities::flowerTypeToDouble(const std::string& type1)
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

Layer Utilities::prodElement(const Layer& tab1, const Layer& tab2)
{
	assert(tab1.size() == tab2.size());
	Layer res;
	for(size_t i(0); i < tab1.size(); ++i){
		res.push_back(tab1[i] * tab2[i]);
	}
return res;
}

double Utilities::prodScal(const Layer& tab1, const Layer& tab2)
{
	assert(tab1.size() == tab2.size());
	double res(0.0);
	for(size_t i(0); i < tab1.size(); ++i){
		res += tab1[i] * tab2[i];
	}
return res;
}

double Utilities::sum(const Layer& tab1)
{
	double res(0.0);
	for(size_t i(0); i < tab1.size(); ++i){
		res+=tab1[i];
	}
	return res;
}

Layer Utilities::vectSum(const Layer& tab1,const Layer& tab2)
{
	assert(tab1.size() == tab2.size());
	Layer res;
	for(size_t i(0); i < tab1.size(); ++i){
		res.push_back(tab1[i] + tab2[i]);			//devrait tout convertir en array pour vitesse, size n'est jamais mod dans programme
	}
	return res;
}

Layer Utilities::vectSub(const Layer& tab1,const Layer& tab2)
{
	assert(tab1.size() == tab2.size());
	Layer res;
	for(size_t i(0); i < tab1.size(); ++i){
		res.push_back(tab1[i] - tab2[i]);			//devrait tout convertir en array pour vitesse, size n'est jamais mod dans programme
	}
	return res;
}

void  Utilities::afficheVect(const Layer& tab)
{
	for(auto i: tab){
		std::cout << i << " ";
	}
	std::cout << std::endl;
}

double  Utilities::sigmoid(double valeur)
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

double Utilities::deriveeSigmoid(double valeur)
{
	return sigmoid(valeur) * (1 - sigmoid(valeur));
}

int Utilities::newInt(std::vector<int>& deja_tires, int data_size)
{
	if(deja_tires.size() >= data_size) deja_tires.clear();
	std::uniform_int_distribution<> dint(0,data_size-1);	//on veut tirer au hasard dans les données pour ne pas tirer les données dans l'ordre
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
