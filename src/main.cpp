#include <iostream>
#include "Network.hpp"

std::ostream& operator<<(std::ostream& sortie, const std::vector<double>& tab)
{
	for(size_t i(0); i < tab.size(); ++i){
		sortie << tab[i] << " ";
	}
	sortie << std::endl;
	
  return sortie;
}

std::ostream& operator<<(std::ostream& sortie, const Matrice& matrice)
{
	for(size_t i(0); i < matrice.size(); ++i){
		sortie << "Layer " << i << " : " << '\t';
		
		for(size_t j(0); j < matrice[i].size(); ++j){
			sortie << matrice[i][j] << '\t';
		}
		sortie << '\n';
	}
	
  return sortie;
}

int main()
{
	try{
		Network net(900, 0.35);
		net.run();
	}
	catch(std::string e){
		std::cerr << "Error: " << e << std::endl;
		return 1;
	}
	catch(...){
		std::cerr << "Undefined error" << std::endl;
		return 2;
	}
return 0;
}



