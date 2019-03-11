#include <cstdio>
#include "mainLogic.h"

int stoneCount = 0;					// 현재까지 둔 돌의 수
vector<vector<int>> gameBoard;		// 게임판 gameBoard[x][y], 0 : 빈 칸, 1 : 내 돌 2 : 적 돌
//vector<point> myLastPlay;			// 지지난 턴에 내가 둔 위치
vector<point> opLastPlay;			// 지난 턴에 적(opponent)이 둔 위치
vector<point> myPlay;				// 우리가 둘 수를 여기에 저장
vector<point> mcCandidates;
unordered_set<point, pair_hash> mainCandidates;
clock_t  startTime;					// 시간 체크


// 그냥 형식상의 메인 함수... 의미 없음.... 테스트용
int fake_main(void)
//int main(void)
{
	startTime = clock();
	mainInit();
	printf("time checker : %d\n", clock() - startTime); startTime = clock();

	//gameBoard[9][9] = MY_STONE;
	//gameBoard[9][10] = OP_STONE;
	//gameBoard[10][10] = OP_STONE;
	//gameBoard[8][10] = MY_STONE;
	//gameBoard[8][8] = MY_STONE;
	//gameBoard[10][9] = OP_STONE;
	//gameBoard[10][8] = OP_STONE;
	//gameBoard[9][7] = MY_STONE;
	//gameBoard[10][7] = MY_STONE;
	//gameBoard[11][8] = OP_STONE;
	//gameBoard[11][7] = OP_STONE;
	//gameBoard[11][10] = MY_STONE;
	//gameBoard[8][7] = MY_STONE;

	gameBoard[9][9] = OP_STONE;
	gameBoard[8][8] = MY_STONE;
	gameBoard[7][9] = MY_STONE;
	gameBoard[10][8] = OP_STONE;
	gameBoard[11][8] = OP_STONE;
	gameBoard[9][7] = MY_STONE;
	gameBoard[6][10] = MY_STONE;
	gameBoard[10][6] = OP_STONE;
	gameBoard[5][11] = OP_STONE;
	gameBoard[8][10] = MY_STONE;
	gameBoard[5][9] = MY_STONE;
	gameBoard[8][5] = OP_STONE;
	gameBoard[8][6] = OP_STONE;
	gameBoard[7][10] = MY_STONE;
	gameBoard[7][11] = MY_STONE;
	gameBoard[4][8] = OP_STONE;
	gameBoard[5][10] = OP_STONE;
	gameBoard[8][12] = OP_STONE;
	// #case 0

	// #case 1
	//gameBoard[8][6] = OP_STONE;
	//gameBoard[8][9] = MY_STONE;

	// # case 2
	//gameBoard[12][7] = MY_STONE;
	//gameBoard[8][11] = OP_STONE;
	//gameBoard[10][11] = OP_STONE;

	// # case 3 
	//gameBoard[8][7] = 0;
	//gameBoard[7][7] = MY_STONE;

	// # case 4
	//gameBoard[5][7] = OP_STONE;
	//gameBoard[12][7] = MY_STONE;

	// #case 5
	//gameBoard[8][6] = MY_STONE;
	//gameBoard[8][9] = MY_STONE;

	unordered_set<point, pair_hash> cand;
	for (int r = 0; r < 19; r++)
		for (int c = 0; c < 19; c++)
			if (!gameBoard[r][c])
				cand.insert(point(r, c));

	printf("left : %d\n", mainChecker(gameBoard, cand, myPlay));
	for (int i = 0; i < myPlay.size(); i++)
	{
		printf("todo : %d %d\n", myPlay[i].xPos, myPlay[i].yPos);
		gameBoard[myPlay[i].xPos][myPlay[i].yPos] = 3;
	}

	printf("time checker : %d\n", clock() - startTime);
	printBoard();
	return 0;
}

// 초기화
void mainInit()
{
	printf("initialize.... ");

	srand(0);
	//srand((unsigned int)time(NULL));
	for (int i = 0; i < 19; i++) gameBoard.push_back(vector<int>(19, 0));

	initChecker();

	gameBoard[9][9] = MY_STONE;

	gameBoard[5][9] = MY_STONE;
	gameBoard[5][11] = MY_STONE;
	gameBoard[6][7] = MY_STONE;
	gameBoard[7][10] = MY_STONE;
	gameBoard[8][4] = MY_STONE;
	gameBoard[8][9] = MY_STONE;
	gameBoard[8][10] = MY_STONE;
	gameBoard[9][9] = MY_STONE;
	gameBoard[10][4] = MY_STONE;
	gameBoard[10][6] = MY_STONE;
	gameBoard[10][8] = MY_STONE;
	gameBoard[10][9] = MY_STONE;
	gameBoard[10][10] = MY_STONE;
	gameBoard[10][11] = MY_STONE;
	gameBoard[11][7] = MY_STONE;
	gameBoard[6][8] = OP_STONE;
	gameBoard[6][10] = OP_STONE;
	gameBoard[7][5] = OP_STONE;
	gameBoard[7][7] = OP_STONE;
	gameBoard[7][9] = OP_STONE;
	gameBoard[8][5] = OP_STONE;
	gameBoard[8][6] = OP_STONE;
	gameBoard[8][7] = OP_STONE;
	gameBoard[8][8] = OP_STONE;
	gameBoard[9][5] = OP_STONE;
	gameBoard[9][7] = OP_STONE;
	gameBoard[10][7] = OP_STONE;
	gameBoard[7][11] = OP_STONE;
	gameBoard[12][6] = OP_STONE;

	auto hasIt = mainCandidates.find(makePoint(9, 9));
	if (hasIt != mainCandidates.end())
		mainCandidates.erase(hasIt);

	updateCandidate(mainCandidates, 9, 9);

	printf("\t ok\n");
}

void updateCandidate(unordered_set<point, pair_hash> cand, int x, int y)
{
	for (int dx = -2; dx <= 2; dx++)
		for (int dy = -2; dy <= 2; dy++)
		{
			int tx = x + dx, ty = y + dy;
			if (tx >= 0 && tx < 19 && ty >=0 && ty < 19 && !gameBoard[tx][ty])
				mainCandidates.insert(makePoint(tx, ty));
		}
}

void saveOpPlay(int opX[], int opY[])
{
	gameBoard[opX[0]][opY[0]] = gameBoard[opX[1]][opY[1]] = OP_STONE;

	for (int i = 0; i < 2; i++)
	{
		int x = opX[i], y = opY[i];
		auto hasIt = mainCandidates.find(makePoint(x, y));
		if (hasIt != mainCandidates.end())
			mainCandidates.erase(hasIt);

		updateCandidate(mainCandidates, x, y);
	}
}

void saveMyPlay(const point &my)
{
	gameBoard[my.xPos][my.yPos] = MY_STONE;
	auto hasIt = mainCandidates.find(my);
	if (hasIt != mainCandidates.end())
		mainCandidates.erase(hasIt);
}

// 메인 로직
void mainLogic(pair<int, int> play[])
{
	preProcess();
	
	// 1~4. 조건 체크, myPlay에 값 넣어서 올 것
	mainChecker(gameBoard, mainCandidates, myPlay);
	if (myPlay.size() == 2)
	{
		play[0] = myPlay[0];
		play[1] = myPlay[1];
		saveMyPlay(myPlay[0]);
		saveMyPlay(myPlay[1]);
		printf("cutted in brute force\n");
		return;
	}
	else if (myPlay.size() == 1)
	{
		saveMyPlay(myPlay[0]);
		printf("pick one in brute force\n");
	}

	// 5. 몬테 카를로 서치
	mainMonteCarlo(gameBoard, mainCandidates, myPlay, startTime);
	printf("end mc search in %d ms\n", clock() - startTime);
	if (myPlay.size() == 2) 
	{
		play[0] = myPlay[0];
		play[1] = myPlay[1];
		saveMyPlay(myPlay[0]);
		saveMyPlay(myPlay[1]);
		return;
	}
	
	// 랜덤으로 비어있는 곳 두기
	// 어차피 의미 없는데 그냥 (0, 0)부터 (18, 18)로 둬도 괜찮음
	saveMyPlay(myPlay[0]);
	saveMyPlay(myPlay[1]);
}

// 로직 시작 전처리, 입력 받는 것을 우리 프로세스에 맞게 여기서 파싱할 것임
void preProcess()
{
	startTime = clock();
	myPlay.clear();
}

void printBoard()
{
	printf("  "); for (int i = 0; i < 10; i++) printf("  "); for (int i = 10; i < 19; i++) printf(" 1"); printf("\n");
	printf("  "); for (int i = 0; i < 10; i++) printf(" %d", i); for (int i = 0; i < 9; i++) printf(" %d", i); printf("\n");
	for (int y = 0; y < 19; y++)
	{
		printf("%2d", y);
		for (int x = 0; x < 19; x++)
		{
			if (gameBoard[x][y] == 0)
				printf(" .");
			else if (gameBoard[x][y] == MY_STONE)
				printf(" O");
			else if (gameBoard[x][y] == OP_STONE)
				printf(" X");
			else
				printf(" T");
		}
		printf("\n");
	}
}

