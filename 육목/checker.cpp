#include "checker.h"

// ����: �� | \  /   
// DX[����][�Ÿ�]
const int DX[4][13] = {
	-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, //0
	0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, //1
	-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
	-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6 }; //3
												   // DY[����][�Ÿ�]
const int DY[4][13] = {
	0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, //0
	-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, //1
	-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
	6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6 }; //3 

												   // 0: ����, 1: ����, 2: \*, 3: /
const int DIR_LEFT_X[4] = { -1 ,  0 , -1 , -1 };
const int DIR_LEFT_Y[4] = { 0 , -1 , -1 , 1 };
const int DIR_RIGHT_X[4] = { 1 ,  0 ,  1 , 1 };
const int DIR_RIGHT_Y[4] = { 0 ,  1 ,  1 , -1 };



//@suggest : ��Ʈ����ũ ���� �� ��
vector<int> checker = vector<int>(1594323, 0);
vector<int> hitCount = vector<int>(1594323);		// ���ƾ� �ϴ� ���� ��
vector<int> appendPoint = vector<int>(1594323);	// �߰��� �־��ϴ� ���� ��ġ (-6~-1, 1~6), ������ 0
												//���� ��Ʈ����ũ ���� a���Ҷ�, hitCount[a]�� appendPoint[a]�� �� ���ϸ鼭 �ϴ°�...

#define R_LEN 38

												//0~12 : ��Ʈ����ũ, 13 : hitCount, 14,15 : appendPoint, 16,17 : blockTwo, 18 : type 
void makeBitmask(vector<int> nums, int index) {
	if (index == 13) { // ��� Ȯ������ ��
					   //���� 
		int bitmask = nums[0] * 531441 +
			nums[1] * 177147 +
			nums[2] * 59049 +
			nums[3] * 19683 +
			nums[4] * 6561 +
			nums[5] * 2187 +
			nums[6] * 729 +
			nums[7] * 243 +
			nums[8] * 81 +
			nums[9] * 27 +
			nums[10] * 9 +
			nums[11] * 3 +
			nums[12] * 1;
		hitCount[bitmask] = nums[13];
		checker[bitmask] = nums[18];

		if (nums[14] != 0) {
			appendPoint[bitmask] = -nums[14];
		}
		else if (nums[15] != 0) {
			appendPoint[bitmask] = nums[15];
		}
		else {
			appendPoint[bitmask] = 0;
		}

		//��� 
		for (vector<int>::iterator iter = nums.begin(); iter != nums.end() - 6; iter++) {
			if (*iter == 1) {
				*iter = 2;
			}
			else if (*iter == 2) {
				*iter = 1;
			}
		}
		nums[18] += 1;
		bitmask = nums[0] * 531441 +
			nums[1] * 177147 +
			nums[2] * 59049 +
			nums[3] * 19683 +
			nums[4] * 6561 +
			nums[5] * 2187 +
			nums[6] * 729 +
			nums[7] * 243 +
			nums[8] * 81 +
			nums[9] * 27 +
			nums[10] * 9 +
			nums[11] * 3 +
			nums[12] * 1;
		hitCount[bitmask] = nums[13];
		checker[bitmask] = nums[18];

		//@todo: deffendPoint[bitmask] & �⺸ ����
		// store append point.
		// if there is no append point, give 10 score.
		if (nums[18] != 2) {
			if (nums[14] != 0) {
				appendPoint[bitmask] = -nums[14];
			}
			else if (nums[15] != 0) {
				appendPoint[bitmask] = nums[15];
			}
			else {
				appendPoint[bitmask] = 0;
			}
		}
		else {
			if (nums[16] != 0) {
				appendPoint[bitmask] = -nums[16];
			}
			else if (nums[17] != 0) {
				appendPoint[bitmask] = nums[17];
			}
			else {
				appendPoint[bitmask] = 0;
			}
		}
		return;
	}

	if (nums[index] == 4) {
		nums[index] = 0;
		makeBitmask(nums, index + 1);
		nums[index] = 1;
		makeBitmask(nums, index + 1);
		nums[index] = 2;
		makeBitmask(nums, index + 1);
	}
	else if (nums[index] == 3) {
		nums[index] = 0;
		makeBitmask(nums, index + 1);
		nums[index] = 2;
		makeBitmask(nums, index + 1);
	}
	else if (nums[index] == 2 || nums[index] == 1 || nums[index] == 0) {
		makeBitmask(nums, index + 1);
	}
	else if (nums[index] == 9) {
		nums[index] = 0;
		makeBitmask(nums, index + 1);
	}
	return;
}

// üĿ �ʱ�ȭ
void initChecker()
{
	// bitmask�� ������ �κ� �ʱ�ȭ
	vector<int> nums;
	char buffer[50];
	//int bitmask;
	FILE *fptr;
	fptr = fopen("newRecord.csv", "r");
	while (fread(buffer, 1, R_LEN, fptr)) {
		for (int i = 0; i < R_LEN; i += 2) {
			nums.push_back(atoi(&buffer[i]));
		}
		//��Ʈ����ũ ������� 
		makeBitmask(nums, 0);
		nums.clear();
	}
	fclose(fptr);
}

// ���� üũ �Լ�
int mainChecker(const vector<vector<int>> &board, const unordered_set<point, pair_hash> &candidate, vector<point> &myPlay)
{
	//clock_t startTime = clock();
	mustToDo(myPlay, 2, board, candidate);					// �θ� �ݵ�� �̱� or �� �θ� �ݵ�� ��
															//printf("time checker for must to do : %d\n", clock() - startTime); startTime = clock();

	size_t left = 2 - myPlay.size();
	if (left) nextToDo(myPlay, left, MY_STONE, board, candidate);	// �θ� �����Ͽ� �̱�
																	//printf("time checker for next to do : %d\n", clock() - startTime); startTime = clock();

	left = 2 - myPlay.size();
	if (left) nextToDo(myPlay, left, OP_STONE, board, candidate);	// �� �θ� ��밡 �����Ͽ� �̱�
																	//printf("time checker for next to do : %d\n", clock() - startTime); startTime = clock();

	return 2 - myPlay.size();
}

// left��ŭ�� ���� ������ �־� �� ���� ���� �ϴ°�, �� �� ���� ��ȯ
int mustToDo(vector<point> &myPlay, int left, const vector<vector<int>> &board, const unordered_set<point, pair_hash> &candidate)
{
	int x, y, oLeft = left, dir;
	int opX[2] = { -1, -1 }, opY[2], opLeft = left;
	int ret = 0;

	for (auto iter = candidate.begin(), end = candidate.end(); iter != end; iter++)
	{
		x = iter->xPos;
		y = iter->yPos;
		for (dir = 0; dir < 4; dir++)
		{
			int bitmask[2] = { 0, 0 };
			//int selected = 0;
			//if(who == OP) fprintf(fp, "find num : %d, dir : %d, who : %d\n", num, dir, who);
			//int dx, dy;
			// ���� Ȯ��
			//dx = x; dy = y;
			//���� �������� bitmask ����� 
			for (int dist = 0, digit = 1; dist <= 12; dist++)
			{
				int nextX = x + DX[dir][dist];
				int nextY = y + DY[dir][dist];
				bitmask[0] *= 3; bitmask[1] *= 3;
				if (!(nextX < 0 || nextY < 0 || nextX>18 || nextY>18)) {
					bitmask[0] += board[nextX][nextY];
					bitmask[1] += board[nextX][nextY];
					//printf("%d", board[nextX][nextY]);
				}
				else {
					bitmask[0] += 1;	// ���� ���(0)�� ���, outOfBoard�� ������ �Ͱ� ����
					bitmask[1] += 2;	// ���� ����(1)�� ���, outOfBoard�� ������ �Ͱ� ����
										//printf("-");
				}
				//if ( checker[bitmask[0]] % 2 == 0 ) selected = 0;
				//else selected = 1;
			}

			if (checker[bitmask[1]] == 1) {
				//printf("must matching!!\n");
				//printf("%d %d\n", bitmask[1], appendPoint[bitmask[1]]);
				myPlay.push_back(makePoint(x, y));
				//prePick(myPlay[0]);
				left--;

				if (left != 0) {
					if (appendPoint[bitmask[1]]<0) {
						myPlay.push_back(makePoint(x - appendPoint[bitmask[1]] * DIR_LEFT_X[dir], y - appendPoint[bitmask[1]] * DIR_LEFT_Y[dir]));
						//prePick(myPlay[1]);
						left--;
					}
					else if (appendPoint[bitmask[1]]>0)
					{
						myPlay.push_back(makePoint(x + appendPoint[bitmask[1]] * DIR_RIGHT_X[dir], y + appendPoint[bitmask[1]] * DIR_RIGHT_Y[dir]));
						//prePick(myPlay[1]);
						left--;
					}
				}
				//printf("check!!!\n");
				return 1;
			}
			// ������ 10�� �̻�, �� �� �� ����, �̸� �α� �� �Ѱ�, ��Ī �Ǵ°�?
			else if ((opLeft == 2 || (opLeft == 1 && !(opX[0] == x && opY[0] == y))) && opLeft > 0
				&& board[x][y] == 0 && checker[bitmask[0]] == 2)
			{
				//printf("%d\n", bitmask[0]);
				if (opX[0] != -1) {
					bool check = false;
					for (dir = 0; dir < 4; dir++)
					{
						int bit = 0;

						for (int dist = 0; dist <= 12; dist++)
						{
							int nextX = x + DX[dir][dist];
							int nextY = y + DY[dir][dist];
							bit *= 3;
							if (!(nextX < 0 || nextY < 0 || nextX>18 || nextY>18)) {
								if (nextX == opX[0] && nextY == opY[0]) {
									bit += 1;
								}
								else {
									bit += board[nextX][nextY];
								}
								//printf("%d", board[nextX][nextY]);
							}
							else {
								bit += 1;	// ���� ���(0)�� ���, outOfBoard�� ������ �Ͱ� ����
											//bitmask[1] += 2;	// ���� ����(1)�� ���, outOfBoard�� ������ �Ͱ� ����
											//printf("-");
							}
							//if ( checker[bitmask[0]] % 2 == 0 ) selected = 0;
							//else selected = 1;
						}

						if (checker[bit] == 2) {
							check = true;
							break;
						}
					}

					if (check == false) { // �ذ�� ���̽��� �Ѿ
						continue;
					}
				}

				//printf("%d %d", x, y);
				//printf("%d %d\n", checker[bitmask[0]], appendPoint[bitmask[0]]);

				opX[oLeft - opLeft] = x;
				opY[oLeft - opLeft] = y;
				opLeft--;

				//if (opLeft == 0) continue;

				// �Ѱ� �� ���ƾ� �ϴ°�?
				if (appendPoint[bitmask[0]] < 0)
				{
					opX[0] = x;
					opY[0] = y;
					opX[1] = x - appendPoint[bitmask[0]] * DIR_LEFT_X[dir];
					opY[1] = y - appendPoint[bitmask[0]] * DIR_LEFT_Y[dir];
					opLeft = 0;
				}
				else if (appendPoint[bitmask[0]] > 0)
				{
					opX[0] = x;
					opY[0] = y;
					opX[1] = x + appendPoint[bitmask[0]] * DIR_RIGHT_X[dir];
					opY[1] = y + appendPoint[bitmask[0]] * DIR_RIGHT_Y[dir];
					opLeft = 0;
				}
				//fprintf(fp, "opLeft : %d\n", opLeft);
			}
		}
	}

	// ���� ���� �� ���µ� ��밡 ���� �� ������
	if (left > 0 && opLeft != oLeft)
	{
		//printf("%d %d\n", opX[0], opY[0]);
		myPlay.push_back(makePoint(opX[0], opY[0]));
		//prePick(makePoint(opX[0], opY[0]));
		left--;
		ret = 2;
		if (left > 0 && opX[1] != -1)
		{
			//printf("%d %d\n", opX[1], opY[1]);
			myPlay.push_back(makePoint(opX[1], opY[1]));
			//prePick(makePoint(opX[1], opY[1]));
			left--;
		}
	}

	return ret;
}


// ���� �Ͽ� �̱� �� �ִ� ��ġ, ex)44
int nextToDo(vector<point> &myPlay, int left, int who, const vector<vector<int>> &board, const unordered_set<point, pair_hash> &candidate)
{
	int x, y, atk, plusAtk, oLeft = left, dir, plusDir, plusNum;
	int strX = -1, strY, strAtk = 0, strDir;

	for (auto iter = candidate.begin(), end = candidate.end(); iter != end; iter++)
	{
		x = iter->xPos;
		y = iter->yPos;
		atk = plusAtk = 0;
		for (dir = 0; dir < 4; dir++)
		{
			int bitmask = 0;
			//int selected = 0;
			//if(who == OP) fprintf(fp, "find num : %d, dir : %d, who : %d\n", num, dir, who);
			//int dx, dy;
			// ���� Ȯ��
			//dx = x; dy = y;
			//���� �������� bitmask ����� 
			for (int dist = 0, digit = 1; dist <= 12; dist++)
			{
				int nextX = iter->xPos + DX[dir][dist];
				int nextY = iter->yPos + DY[dir][dist];
				bitmask *= 3;
				if (!(nextX < 0 || nextY < 0 || nextX>18 || nextY>18)) {
					bitmask += board[nextX][nextY];
					//printf("%d", board[nextX][nextY]);
				}
				else {
					bitmask += (who % 2) + 1;
					//bitmask[0] += 1;	// ���� ���(0)�� ���, outOfBoard�� ������ �Ͱ� ����
					//bitmask[1] += 2;	// ���� ����(1)�� ���, outOfBoard�� ������ �Ͱ� ����
					//printf("-");
				}
				//if ( checker[bitmask[0]] % 2 == 0 ) selected = 0;
				//else selected = 1;
			}

			if ( who == MY_STONE && checker[bitmask] == 3) {
				atk += hitCount[bitmask];
			}
			else if (who == OP_STONE && checker[bitmask] == 4) {
				atk += hitCount[bitmask];
			}

			//if(oLeft == 0) continue; //���� �뵵�ΰ���?? 

			if (who == MY_STONE && checker[bitmask] == 5 && plusAtk < hitCount[bitmask]) {
				plusAtk = hitCount[bitmask];
				plusDir = dir;
				plusNum = bitmask;
			}
			else if (who == OP_STONE && checker[bitmask] == 6 && plusAtk < hitCount[bitmask]) {
				plusAtk = hitCount[bitmask];
				plusDir = dir;
				plusNum = bitmask;
			}
		}
		//if(atk)	printf("at %d, %d / test : %d %d\n", x, y, atk, plusAtk);
		// ��� ���� ��ȸ ��, 
		// �ѹ� �ּ� ��������Ʈ 3 �̻�
		if (who == MY_STONE)
		{
			if (atk > 2) {
				myPlay.push_back(makePoint(x, y));
				//prePick(makePoint(x, y));
				return 3;
			}
			// 2�� �� �� �ְ�, �⺻ ��������Ʈ�� �߰� ��������Ʈ�� ���� 3�̻�
			else if (oLeft == 2 && atk + plusAtk > 2)
			{
				myPlay.push_back(makePoint(x, y));
				//prePick(makePoint(x, y));

				if (appendPoint[plusNum] < 0)
				{
					myPlay.push_back(makePoint(x - appendPoint[plusNum] * DIR_LEFT_X[plusDir], y - appendPoint[plusNum] * DIR_LEFT_Y[plusDir]));
				}
				else
				{
					myPlay.push_back(makePoint(x + appendPoint[plusNum] * DIR_RIGHT_X[plusDir], y + appendPoint[plusNum] * DIR_RIGHT_Y[plusDir]));
				}
				return 3;
			}

			if (left == 2)
			{
				if (strX != -1 && atk + strAtk > 2)
				{
					if (noSameLine(x, y, strX, strY, board, who))
					{
						myPlay.push_back(makePoint(x, y));
						myPlay.push_back(makePoint(strX, strY));
						return 3;
					}
				}
				if (atk >= strAtk)
				{
					strX = x;
					strY = y;
					strAtk = atk;
				}
			}
		}
		//// �� �������� ������ (who = OP)
		else
		{
			if (atk + plusAtk > 2)
			{
				myPlay.push_back(makePoint(x, y));
				//prePick(makePoint(x, y));
				left--;
				if (left == 0) return 3;
			}

			if (strX != -1 && noSameLine(x, y, strX, strY, board, who))
			{
				if (atk == 2)
					myPlay.push_back(makePoint(x, y));
				if (strAtk == 2)
					myPlay.push_back(makePoint(strX, strY));
				return 3;
			}

			if (atk >= strAtk)
			{
				strX = x;
				strY = y;
				strAtk = atk;
			}
		}
		//else if (strX != -1 && atk + strAtk > 2)
		//{
		//	if (atk > strAtk)
		//	{
		//		myPlay.push_back(makePoint(x, y));
		//		prePick(makePoint(x, y));
		//	}
		//	else
		//	{
		//		myPlay.push_back(makePoint(strX, strY));
		//		prePick(makePoint(strX, strY));
		//	}
		//	left--;
		//	if (left == 0) return oLeft;
		//}
	}

	return 0;
}

bool noSameLine(int x1, int y1, int x2, int y2, const vector<vector<int>> &board, int who)
{
	//printf("check %d %d, %d %d\n", x1, y1, x2, y2);
	// ���η� ����
	if (x1 == x2)
	{
		int d = y2 - y1 > 0 ? 1 : -1;
		int serial = 0, maxSerial = 0;
		for (int ty = y1 - (2 * d); ty != x2 + (3 * d); ty += d)
		{
			if (ty < 0 || ty > 18)
				serial = 0;
			else if (!board[x1][ty] && board[x1][ty] != who)
				serial = 0;
			else if (board[x1][ty] == who)
			{
				serial++;
				maxSerial = maxSerial < serial ? serial : maxSerial;
			}
		}
		if (maxSerial > 1) return false;
	}
	// ���η� ����
	else if (y1 == y2)
	{
		int d = x2 - x1 > 0 ? 1 : -1;
		int serial = 0, maxSerial = 0;
		for (int tx = x1 - (2 * d); tx != x2 + (3 * d); tx += d)
		{
			if (tx < 0 || tx > 18)
				serial = 0;
			else if (!board[tx][y1] && board[tx][y1] != who)
				serial = 0;
			else if (board[tx][y1] == who)
			{
				serial++;
				maxSerial = maxSerial < serial ? serial : maxSerial;
			}
		}
		if (maxSerial > 1) return false;
	}
	else if (x1 - x2 == y1 - y2 || x1 - x2 == y2 - y1)
	{
		int dx = x2 - x1 > 0 ? 1 : -1;
		int dy = y2 - y1 > 0 ? 1 : -1;
		int serial = 0, maxSerial = 0;
		for (int tx = x1 - (2 * dx), ty = y1 - (2 * dx); tx != x2 + (3 * dx); tx += dx, ty += dy)
		{
			//printf("\tin %d %d... %d\n", tx, ty, board[tx][ty]);
			if (tx < 0 || tx > 18 || ty < 0 || ty > 18)
				serial = 0;
			else if (!board[tx][ty] && board[tx][ty] != who)
				serial = 0;
			else if (board[tx][ty] == who)
			{
				serial++;
				maxSerial = maxSerial < serial ? serial : maxSerial;
			}
		}
		if (maxSerial > 1) return false;
	}
	// ���� �� ��
	//printf("\t\tpass!!\n");
	return true;
}

/*
// �⺸ num�� x, y, dir�� ��Ī�Ǵ°�?
// search[x+1][y+1]�� ���� ������ �����ϰ� ��Ī�غ��� ��. ���� ���� 2���� ū��? �� Ȯ��
// num�� Form*�� �Ѱ��༭ �ϴ°� �¾Ҵµ�... ��ĥ �ð��� �����ϴ�...
bool match(int num, int x, int y, int dir, int who, int target, const vector<vector<int>> &board)
{
int bitmask[2] = { 0 }, selected = 0;
//if(who == OP) fprintf(fp, "find num : %d, dir : %d, who : %d\n", num, dir, who);
int i, dx, dy;
// ���� Ȯ��
dx = x; dy = y;

bitmask[0] = 0; bitmask[1] = 0;
for (int dist = 0, digit = 1; dist <= 12; dist++)
{
dx += DIR_LEFT_X[dir];
dy += DIR_LEFT_Y[dir];
bitmask[0] *= 3; bitmask[1] *= 3;
if (!(dx < 0 || dy < 0 || dx>18 || dy>18)) {
bitmask[0] += board[nextX][nextY];
bitmask[1] += board[nextX][nextY];
//printf("%d", board[nextX][nextY]);
}
else {
bitmask[0] += 1;	// ���� ���(0)�� ���, outOfBoard�� ������ �Ͱ� ����
bitmask[1] += 2;	// ���� ����(1)�� ���, outOfBoard�� ������ �Ͱ� ����
//printf("-");
}
if ( checker[bitmask[0]] % 2 == 0 ) selected = 0;
else selected = 1;
}

for (i = 0; i < forms[target][num].leftLen; i++)
{
dx += DIR_LEFT_X[dir];
dy += DIR_LEFT_Y[dir];

if (dx < 0 || dx > 18 || dy < 0 || dy > 18) return false;
else if (forms[target][num].left[i] == 3 && board[dx][dy] == who) return false;		// �� ���� �ƴϸ� �� : �� ����
else if (forms[target][num].left[i] == 1 && board[dx][dy] != who) return false;		// �� ���̰ų� ��ĭ�̿��� ��
else if (forms[target][num].left[i] == 0 && board[dx][dy] != 0) return false;		// �� ���̰ų� ��ĭ�̿��� ��
}
// ������ Ȯ��
dx = x; dy = y;
for (i = 0; i < forms[target][num].rightLen; i++)
{
dx += DIR_RIGHT_X[dir];
dy += DIR_RIGHT_Y[dir];

if (dx < 0 || dx > 18 || dy < 0 || dy > 18) return false;
else if (forms[target][num].right[i] == 3 && board[dx][dy] == who) return false;
else if (forms[target][num].right[i] == 1 && board[dx][dy] != who) return false;		// �� ���̰ų� ��ĭ�̿��� ��
else if (forms[target][num].right[i] == 0 && board[dx][dy] != 0) return false;		// �� ���̰ų� ��ĭ�̿��� ��
}

//printf("%d match it!\n", target);
// ���� ��
return true;
}

void prePick(point pick)
{
printf("pick : %d %d\n", pick.xPos, pick.yPos);
}
*/
