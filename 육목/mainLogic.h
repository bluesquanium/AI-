#pragma once

#include "util.h"
#include "checker.h"
#include "monteCarlo.h"

void mainInit();
void mainLogic(pair<int, int> play[]);
void preProcess();
void printBoard();
void saveOpPlay(int opX[], int opY[]);
void updateCandidate(unordered_set<point, pair_hash> cand, int tx, int ty);
void saveMyPlay(const point &my);