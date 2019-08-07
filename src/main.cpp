#include <iostream>
#include "Network.hpp"

int main()
{
	try{
		Network net(300, 0.35);
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



