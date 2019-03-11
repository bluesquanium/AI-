#pragma once
#include "util.h"

// 초기화 함수
void initChecker();

//bool match(int num, int x, int y, int dir, int who, int target, const vector<vector<int>> &board);
int mustToDo(vector<point> &myPlay, int left, const vector<vector<int>> &board, const unordered_set<point, pair_hash> &candidate);
int nextToDo(vector<point> &myPlay, int left, int who, const vector<vector<int>> &board, const unordered_set<point, pair_hash> &candidate);
bool noSameLine(int x1, int y1, int x2, int y2, const vector<vector<int>> &board, int who);
// 메인 체크 함수
int mainChecker(const vector<vector<int>> &board, 
	const unordered_set<point, pair_hash> &candidate, vector<point> &myPlay);
