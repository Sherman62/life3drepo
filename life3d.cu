/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号: SA24011134
 * 姓名: 李金优
 * 邮箱: ljinyou@mail.ustc.edu.cn
 ------------------------------------------------*/

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include "cuda.h" 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define AT(x, y, z) universe[(x) * N * N + (y) * N + z]

using namespace std;
//using std::cin, std::cout, std::endl;
//using std::ifstream, std::ofstream;

// 存活细胞数
int population(int N, char *universe)
{
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += universe[i];
    return result;
}


void print_universe(int N, char *universe)
{
 
    if (N > 32)
        return;
    for (int x = 0; x < N; x++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int z = 0; z < N; z++)
            {
                if (AT(x, y, z))
                    cout << "O ";
                else
                    cout << "* ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "population: " << population(N, universe) << endl;
}

//*********************************** 添加global
__global__ 
void life3d_run(int N, int cbsize,char *universe,char*next, int T)
{
    
   
    extern __shared__ int cubeBlock[];

    //线程在总体中的下标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    //每个线程在线程块中的下标（相当于threadIdx.x,y,z）
    int cbx = (x%cbsize) + 1,cby = (y%cbsize) + 1,cbz = (z% cbsize) + 1;
    //每个block最下角的线程下标为：
    // （(x/cbsize*cbsize)， (y/cbsize*cbsize)， (z/cbsize*cbsize)）
    //之后再次基础上对六个面进行赋值。注意该点需要各个维度减去1才能得到cblock的位置

    int alive = 0,ans;
    //对坐标都迁移过了，不需要额外加上base
    // int cbbase = (cbsize+2)*(cbsize+2)+(cbsize+2)+1;
    
    //把下面当成通过坐标计算，而不是一维计算。
    int cbIdx = cbx*(cbsize+2)*(cbsize+2) + cby*(cbsize+2) + cbz;
    //每个线程对对应位置赋值一次，即进行一次访存。
    cubeBlock[cbIdx] = AT(x,y,z);

    //先给cubeblock外围上下左右前后六面边填充填充边缘值
    //本来想借助cbx列举几个数，发现之前设置的cbx最小值为1.
    //没有做到六个面之间重合的边只读取一次内存，可以继续改进。
    if(cbx == 1 && cby == 1 && cbz == 1){
        //找 x= 0 ,yz平面 
        int stx = (x/cbsize*cbsize) - 1;
        int sty = (y/cbsize*cbsize) - 1;
        int stz = (z/cbsize*cbsize) - 1;

        int dx = 0;
        for(int dy = 0;dy<=cbsize+1;dy++){
                for(int dz = 0;dz<= cbsize+1;dz++){
                
                int offset = dx * (cbsize+2) *(cbsize+2) + dy * (cbsize+2) + dz;
                    //同时考虑在universe中出界的问题
                cubeBlock[offset] = AT((stx+0+N)%N,(sty+dy+N)%N,(stz+dz+N)%N);
                }
        }
    }
    else if(cbx == 1 && cby == 1 && cbz == 2){
        //找 x= cbsize+1 ,yz平面 
        int stx = (x/cbsize*cbsize) - 1 + (cbsize+1);
        int sty = (y/cbsize*cbsize) - 1;
        int stz = (z/cbsize*cbsize) - 1;

        int dx = cbsize+1;
        for(int dy = 0;dy<=cbsize+1;dy++){
                for(int dz = 0;dz<= cbsize+1;dz++){
                
                int offset = dx * (cbsize+2) *(cbsize+2) + dy * (cbsize+2) + dz;
                    //同时考虑在universe中出界的问题
                cubeBlock[offset] = AT((stx+0+N)%N,(sty+dy+N)%N,(stz+dz+N)%N);
                }
        }  
    }
    else if(cbx == 2 && cby == 2 && cbz == 1){
            //令y = 0 ,变动xz平面
        int stx = (x/cbsize*cbsize) - 1;
        int sty = (y/cbsize*cbsize) - 1;
        int stz = (z/cbsize*cbsize) - 1;

        int dy = 0;
        for(int dx = 0;dx<=cbsize+1;dx++){
                for(int dz = 0;dz<= cbsize+1;dz++){
                
                int offset = dx * (cbsize+2) *(cbsize+2) + dy * (cbsize+2) + dz;
                    //同时考虑在universe中出界的问题
                cubeBlock[offset] = AT((stx+dx+N)%N,(sty+0+N)%N,(stz+dz+N)%N);
                }
        }            
    }  
    else if(cbx == 2 && cby == 2 && cbz == 2){
            //令y = cubesize+1 ,变动xz平面
    
        int stx = (x/cbsize*cbsize) - 1;
        int sty = (y/cbsize*cbsize) - 1 + (cbsize+1);
        int stz = (z/cbsize*cbsize) - 1;

        int dy = cbsize+1;
        for(int dx = 0;dx<=cbsize+1;dx++){
                for(int dz = 0;dz<= cbsize+1;dz++){
                
                int offset = dx * (cbsize+2) *(cbsize+2) + dy * (cbsize+2) + dz;
                    //同时考虑在universe中出界的问题
                cubeBlock[offset] = AT((stx+dx+N)%N,(sty+0+N)%N,(stz+dz+N)%N);
                }
        }         
    } 
    else if(cbx == 3 && cby == 3 && cbz == 1){
            //令z = 0 ,变动xy平面
   
        int stx = (x/cbsize*cbsize) - 1;
        int sty = (y/cbsize*cbsize) - 1;
        int stz = (z/cbsize*cbsize) - 1;

        int dz = 0;
        for(int dx = 0;dx<=cbsize+1;dx++){
                for(int dy = 0;dy<= cbsize+1;dy++){
                
                int offset = dx * (cbsize+2) *(cbsize+2) + dy * (cbsize+2) + dz;
                    //同时考虑在universe中出界的问题
                cubeBlock[offset] = AT((stx+dx+N)%N,(sty+dy+N)%N,(stz+0+N)%N);
                }
        }      
    } 
    else if(cbx == 3 && cby == 3 && cbz == 2){
            //令z = cbsize+1 ,变动xy平面
 
        int stx = (x/cbsize*cbsize) - 1;
        int sty = (y/cbsize*cbsize) - 1;
        int stz = (z/cbsize*cbsize) - 1 + (cbsize + 1);

        int dz = cbsize+1;
        for(int dx = 0;dx<=cbsize+1;dx++){
                for(int dy = 0;dy<= cbsize+1;dy++){
                
                int offset = dx * (cbsize+2) *(cbsize+2) + dy * (cbsize+2) + dz;
                    //同时考虑在universe中出界的问题
                cubeBlock[offset] = AT((stx+dx+N)%N,(sty+dy+N)%N,(stz+0+N)%N);
                }
        }              
    } 
    __syncthreads();

    //超出cube后需要从外界取数，还需要判断是否超出最大者，最大者需要取模。
    //6+12+8 = 26，一共26种情况
    // for (int dx = -1; dx <= 1; dx++)
    //         for (int dy = -1; dy <= 1; dy++)
    //             for (int dz = -1; dz <= 1; dz++)
    //             {
    //                 if (dx == 0 && dy == 0 && dz == 0)
    //                     continue;
    //                 int nx = cbx + dx;
    //                 int ny = cby + dy;
    //                 int nz = cbz + dz;
    //                 alive += cubeBlock[nx*(cbsize+2)*(cbsize+2)+ny*(cbsize+2)+nz];
    //             }
    alive += cubeBlock[cbIdx+1];
    alive += cubeBlock[cbIdx-1];
    alive += cubeBlock[cbIdx+(cbsize+2)];
    alive += cubeBlock[cbIdx-(cbsize+2)];
    alive += cubeBlock[cbIdx+(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx-(cbsize+2)*(cbsize+2)];

    alive += cubeBlock[cbIdx+1+(cbsize+2)];
    alive += cubeBlock[cbIdx+1-(cbsize+2)];
    alive += cubeBlock[cbIdx+1+(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx+1-(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx-1+(cbsize+2)];
    alive += cubeBlock[cbIdx-1-(cbsize+2)];
    alive += cubeBlock[cbIdx-1+(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx-1-(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx+(cbsize+2)+(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx-(cbsize+2)+(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx+(cbsize+2)-(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx-(cbsize+2)-(cbsize+2)*(cbsize+2)];

    alive += cubeBlock[cbIdx+1+(cbsize+2)+(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx+1+(cbsize+2)-(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx+1-(cbsize+2)+(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx-1+(cbsize+2)+(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx+1-(cbsize+2)-(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx-1-(cbsize+2)+(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx-1+(cbsize+2)-(cbsize+2)*(cbsize+2)];
    alive += cubeBlock[cbIdx-1-(cbsize+2)-(cbsize+2)*(cbsize+2)];
    
    if (cubeBlock[cbIdx] && (alive < 5 || alive > 7))
        next[x * N * N + y * N + z] = 0;
    else if (!cubeBlock[cbIdx] && alive == 6)
        next[x * N * N + y * N + z] = 1;
    else
        next[x * N * N + y * N + z] = cubeBlock[cbIdx];
   
}



void check(int N, char *universe, int T)
{
    char *next = (char *)malloc(N * N * N);
    for (int t = 0; t < T; t++)
    {
        // outerloop: iter universe
        for (int x = 0; x < N; x++)
            for (int y = 0; y < N; y++)
                for (int z = 0; z < N; z++)
                {
                    // inner loop: stencil
                    int alive = 0;
                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dz = -1; dz <= 1; dz++)
                            {
                                if (dx == 0 && dy == 0 && dz == 0)
                                    continue;
                                int nx = (x + dx + N) % N;
                                int ny = (y + dy + N) % N;
                                int nz = (z + dz + N) % N;
                                alive += AT(nx, ny, nz);
                            }
                    if (AT(x, y, z) && (alive < 5 || alive > 7))
                        next[x * N * N + y * N + z] = 0;
                    else if (!AT(x, y, z) && alive == 6)
                        next[x * N * N + y * N + z] = 1;
                    else
                        next[x * N * N + y * N + z] = AT(x, y, z);
                }
        memcpy(universe, next, N * N * N);
    }
    free(next);
}
// 读取输入文件
void read_file(char *input_file, char *buffer)
{
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        cout << "Error: Could not open file " << input_file << std::endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer, file_size))
    {
        std::cerr << "Error: Could not read file " << input_file << std::endl;
        exit(1);
    }
    file.close();
}

// 写入输出文件
void write_file(char *output_file, char *buffer, int N)
{
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        cout << "Error: Could not open file " << output_file << std::endl;
        exit(1);
    }
    file.write(buffer, N * N * N);
    file.close();
}

int main(int argc, char **argv)
{
    // cmd args
    if (argc < 5)
    {
        cout << "usage: ./life3d N T input output" << endl;
        return 1;
    }
    int N = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);
    char *input_file = argv[3];
    char *output_file = argv[4];


    int cbSize = 3 , bn = (N+cbSize-1)/cbSize;
    char *universe = (char *)malloc(N * N * N);
    char *d_universe,*next;
    char*ans_universe = (char *)malloc(N * N * N);
    read_file(input_file, universe);

    memcpy(ans_universe, universe, N * N * N);
    check(N, ans_universe, T);
    int right_pop = population(N, ans_universe);

    int start_pop = population(N, universe);
    auto start_time = std::chrono::high_resolution_clock::now();
    

    //*********************************** 
    cudaMalloc((void**)&d_universe,N*N*N);
    cudaMemcpy((void*)d_universe,(void*)universe,N*N*N,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&next,N*N*N);
    cudaMemcpy((void*)next,(void*)universe,N*N*N,cudaMemcpyHostToDevice);
    


    int cbsize = 4;  // block size
    int gsize = (N+cbsize-1)/cbsize;
    dim3 blockSize(cbsize,cbsize,cbsize);
    dim3 gridSize(gsize,gsize,gsize);


    for(int i = 0;i<T;i++)
    {
         life3d_run<<<gridSize,blockSize,(cbsize+2)*(cbsize+2)*(cbsize+2)*sizeof(int)>>>(N, cbsize,d_universe,next, T);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();

        if(err!= cudaSuccess){
            printf("CUDA Error1: %s\n",cudaGetErrorString(err));
        }
        cudaMemcpy((void*)d_universe,(void*)next,N*N*N,cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy((void*)universe,(void*)d_universe,N*N*N,cudaMemcpyDeviceToHost);

    //*********************************** 

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    int final_pop = population(N, universe);
    write_file(output_file, universe, N);

    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    cout << "right population: " << right_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cell per sec: " << T / time * N * N * N << endl;

    //*********************************** */
    cudaFree(d_universe);

    free(universe);
    return 0;
}
