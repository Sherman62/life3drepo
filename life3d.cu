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

using std::cin, std::cout, std::endl;
using std::ifstream, std::ofstream;

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
void life3d_run(int N, int cbsize,char *universe,char *next, int T)
{
    
    // char *next = (char *)malloc(N * N * N);
`   //原数据最少也是3*3*3，那么假设这里file开到5*5*5，能够计算得到3*3的数据。
     //原数据边数是N，那么分的块数应该是N/5向上取整。
    extern __shared__ int cubeBlock[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int cbx = x%3,cby = y%3,cyz = z%3;
    int center = (x/3*3)*blockDim.x + (y/3*3)*blockDim.y + (z/3*3)*blockDim.z;

    int alive = 0,ans;
    for (int t = 0; t < T; t++)
    {
        //因为同步函数和只有一个block的原因能够保持数据正确性
        //循环展开的同时使用tile减少对内存读入。
        //选择在每次循环中将next数组读入共享内存。
        //想把三维线程号对应映射到一个中心化后的位置。
        int base = (cbsize+2)*(cbsize+2)+(cbsize+2)+1; 
        int cur = base + cbx*(cbsize+2)*(cbsize+2) + cby*(cbsize+2) + cbz;
        cubeBlock[cur] = universe[cur];
        //赋值结束需要统计临近个数。对于所有需要的数在当前空间的才能缩短时间。

        //先给cubeblock上下六个边填充填充边缘值
        if(cbx == 0 && cby == 0 && cbz == 0){
            //找 x= 0 ,变动yz平面
            //该面的中心距离cubeblock的中心相距(cbsize)/2 个 (cbsize+2)*(cbsize+2)
            //这里默认cubesize是奇数了,且该距离可加可减去。
            //坏了，就算是cbsize是奇数，外面的层cbsize+2也是奇数
            int cubeface_center = center - (cbsize)/2*(cbsize+2)*(cbsize+2);
            int blockface_center = cubeface_center - (cbsize+2)*(cbsize+2);
            for(int dy = -(cbsize+2)/2;dy<=(cbsize+2)/2;dy++){
                 for(int dz = -(cbsize+2)/2;dz<=(cbsize+2)/2;dz++){
                    //   int offset = dx * (cbsize+2) *(cbsize+2) + dy * (cbsize+2) + dz;
                    int offset =  dy * (cbsize+2) + dz;
                      //同时考虑在universe中出界的问题
                      cubeBlock[cubeface_center+offset] = universe[(blockface_center +offset + N)%N];
                 }
            }
        }
        else if(cbx == 0 && cby == 0 && cbz == 1){
              //令x = cbsize-1 ,变动yz平面
            //该面的中心距离cubeblock的中心相距(cbsize)/2 个 (cbsize+2)*(cbsize+2)
            //这里默认cubesize是奇数了,且该距离可加可减去。
            //坏了，就算是cbsize是奇数，外面的层cbsize+2也是奇数
            int cubeface_center = center + (cbsize)/2*(cbsize+2)*(cbsize+2);
            int blockface_center = cubeface_center + (cbsize+2)*(cbsize+2);
            for(int dy = -(cbsize+2)/2;dy<=(cbsize+2)/2;dy++){
                 for(int dz = -(cbsize+2)/2;dz<=(cbsize+2)/2;dz++){
                      int offset =  dy * (cbsize+2) + dz;
                      //同时考虑在universe中出界的问题
                      cubeBlock[cubeface_center+offset] = universe[(blockface_center +offset + N)%N];
                 }
            }           
        }
        else if(cbx == 0 && cby == 1 && cbz == 0){
              //令y = 0 ,变动xz平面
            //该面的中心距离cubeblock的中心相距(cbsize)/2 个 (cbsize+2)
            //这里默认cubesize是奇数了,且该距离可加可减去。
            //坏了，就算是cbsize是奇数，外面的层cbsize+2也是奇数
            int cubeface_center = center - (cbsize)/2*(cbsize+2);
            int blockface_center = cubeface_center - (cbsize+2);
            for(int dx = -(cbsize+2)/2;dx<=(cbsize+2)/2;dx++){
                 for(int dz = -(cbsize+2)/2;dz<=(cbsize+2)/2;dz++){
                      int offset = dx * (cbsize+2) *(cbsize+2) + dz;
                      //同时考虑在universe中出界的问题
                      cubeBlock[cubeface_center+offset] = universe[(blockface_center +offset + N)%N];
                 }
            }           
        }  
        else if(cbx == 0 && cby == 1 && cbz == 1){
              //令y = cubesize-1 ,变动xz平面
            //该面的中心距离cubeblock的中心相距(cbsize)/2 个 (cbsize+2)
            //这里默认cubesize是奇数了,且该距离可加可减去。
            //坏了，就算是cbsize是奇数，外面的层cbsize+2也是奇数
            int cubeface_center = center + (cbsize)/2*(cbsize+2);
            int blockface_center = cubeface_center + (cbsize+2);
            for(int dx = -(cbsize+2)/2;dx<=(cbsize+2)/2;dx++){
                 for(int dz = -(cbsize+2)/2;dz<=(cbsize+2)/2;dz++){
                      int offset = dx * (cbsize+2) *(cbsize+2) + dz;
                      //同时考虑在universe中出界的问题
                      cubeBlock[cubeface_center+offset] = universe[(blockface_center +offset + N)%N];
                 }
            }           
        } 
        else if(cbx == 1 && cby == 0 && cbz == 0){
              //令z = cubesize-1 ,变动xy平面
            //该面的中心距离cubeblock的中心相距(cbsize)/2 个 1
            //这里默认cubesize是奇数了,且该距离可加可减去。
            //坏了，就算是cbsize是奇数，外面的层cbsize+2也是奇数
            int cubeface_center = center + (cbsize)/2*1;
            int blockface_center = cubeface_center + 1;
            for(int dx = -(cbsize+2)/2;dx<=(cbsize+2)/2;dx++){
                 for(int dy = -(cbsize+2)/2;dy<=(cbsize+2)/2;dy++){
                      int offset = dx * (cbsize+2) *(cbsize+2) + dy*(cbsize+2);
                      //同时考虑在universe中出界的问题
                      cubeBlock[cubeface_center+offset] = universe[(blockface_center +offset + N)%N];
                 }
            }           
        } 
        else if(cbx == 1 && cby == 0 && cbz == 1){
              //令z = 0 ,变动xy平面
            //该面的中心距离cubeblock的中心相距(cbsize)/2 个 1
            //这里默认cubesize是奇数了,且该距离可加可减去。
            //坏了，就算是cbsize是奇数，外面的层cbsize+2也是奇数
            int cubeface_center = center - (cbsize)/2*1;
            int blockface_center = cubeface_center - 1;
            for(int dx = -(cbsize+2)/2;dx<=(cbsize+2)/2;dx++){
                 for(int dy = -(cbsize+2)/2;dy<=(cbsize+2)/2;dy++){
                      int offset = dx * (cbsize+2) *(cbsize+2) + dy*(cbsize+2);
                      //同时考虑在universe中出界的问题
                      cubeBlock[cubeface_center+offset] = universe[(blockface_center +offset + N)%N];
                 }
            }           
        } 
        __syncthreads();

        //超出cube后需要从外界取数，还需要判断是否超出最大者，最大者需要取模。
        //6+12+8 = 26，一共26种情况
        alive += cubeBlock[cur+1];
        alive += cubeBlock[cur-1];
        alive += cubeBlock[cur+(cbsize+2)];
        alive += cubeBlock[cur-(cbsize+2)];
        alive += cubeBlock[cur+(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur-(cbsize+2)*(cbsize+2)];

        alive += cubeBlock[cur+1+(cbsize+2)];
        alive += cubeBlock[cur+1-(cbsize+2)];
        alive += cubeBlock[cur+1+(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur+1-(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur-1+(cbsize+2)];
        alive += cubeBlock[cur-1-(cbsize+2)];
        alive += cubeBlock[cur-1+(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur-1-(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur+(cbsize+2)+(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur-(cbsize+2)+(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur+(cbsize+2)-(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur-(cbsize+2)-(cbsize+2)*(cbsize+2)];

        alive += cubeBlock[cur+1+(cbsize+2)+(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur+1+(cbsize+2)-(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur+1-(cbsize+2)+(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur-1+(cbsize+2)+(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur+1-(cbsize+2)-(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur-1-(cbsize+2)+(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur-1+(cbsize+2)-(cbsize+2)*(cbsize+2)];
        alive += cubeBlock[cur-1-(cbsize+2)-(cbsize+2)*(cbsize+2)];
        
        if (cubeBlock[cur] && (alive < 5 || alive > 7))
            ans = 0;
        else if (!cubeBlock[cur] && alive == 6)
            ans = 1;
        else
            ans = cubeBlock[cur];

        universe[cur] = ans;
        __syncthreads();
        
    }


    // for (int t = 0; t < T; t++)
    // {
        
    //     for (int x = 0; x < N; x++)
    //         for (int y = 0; y < N; y++)
    //             for (int z = 0; z < N; z++)
    //             {
                    
    //                 int alive = 0;
    //                 for (int dx = -1; dx <= 1; dx++)
    //                     for (int dy = -1; dy <= 1; dy++)
    //                         for (int dz = -1; dz <= 1; dz++)
    //                         {
    //                             if (dx == 0 && dy == 0 && dz == 0)
    //                                 continue;
    //                             int nx = (x + dx + N) % N;
    //                             int ny = (y + dy + N) % N;
    //                             int nz = (z + dz + N) % N;
    //                             alive += AT(nx, ny, nz);
    //                         }
    //                 if (AT(x, y, z) && (alive < 5 || alive > 7))
    //                     next[x * N * N + y * N + z] = 0;
    //                 else if (!AT(x, y, z) && alive == 6)
    //                     next[x * N * N + y * N + z] = 1;
    //                 else
    //                     next[x * N * N + y * N + z] = AT(x, y, z);
    //             }
    //     //memcpy(universe, next, N * N * N);
    //     cudaMemcpy((void*)universe,(void*)next,N*N*N,cudaMemcpyDeviceToDevice);
    // }
    // free(next);
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
    read_file(input_file, universe);

    int start_pop = population(N, universe);
    auto start_time = std::chrono::high_resolution_clock::now();
    

    //*********************************** 

    cudaMalloc((void*)&d_universe,N*N*N);
    cudaMemcpy((void*)d_universe,(void*)universe,N*N*N,cudaMemcpyHostToDevice);
    cudaMalloc((void*)&d_next,N*N*N);
    
    int total = N*N*N;
    dim3 blockSize(cbSize,cbSize,cbSize);
    dim3 gridSize();

    life3d_run<<<gridSize,blockSize,(cbSize+2)*(cbSize+2)*(cbSize+2)*sizeof(int)>>>(N, cbsize,universe, next,T);

    cudaMemcpy((void*)universe,(void*)d_universe,N*N*N,cudaMemcpyDeviceToHost)

    //*********************************** 

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    int final_pop = population(N, universe);
    write_file(output_file, universe, N);

    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cell per sec: " << T / time * N * N * N << endl;

    //*********************************** */
    cudaFree(d_universe);

    free(universe);
    return 0;
}
