#include <torch/torch.h>
#include <iostream>

using namespace std;

int main(int argc, char* argv[]){
    torch::Tensor tensor = torch::randn({2, 3});

    cout << tensor << endl;
    cout << "Torch is included properly" << endl;
    return 0;
}