// MLCG.cc
/* 
Copyright (C) 1989 Free Software Foundation

This file is part of the GNU C++ Library.  This library is free
software; you can redistribute it and/or modify it under the terms of
the GNU Library General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your
option) any later version.  This library is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the GNU Library General Public License for more details.
You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the Free Software
Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include "MLCG.h"
//
//	SEED_TABLE_SIZE must be a power of 2
//


#define SEED_TABLE_SIZE 32

static int seedTable[SEED_TABLE_SIZE] = {
static_cast<int>(0xbdcc47e5), 0x54aea45d, static_cast<int>(0xec0df859), static_cast<int>(0xda84637b),
static_cast<int>(0xc8c6cb4f), 0x35574b01, 0x28260b7d, 0x0d07fdbf,
static_cast<int>(0x9faaeeb0), 0x613dd169, 0x5ce2d818, static_cast<int>(0x85b9e706),
static_cast<int>(0xab2469db), static_cast<int>(0xda02b0dc), 0x45c60d6e, static_cast<int>(0xffe49d10),
0x7224fea3, static_cast<int>(0xf9684fc9), static_cast<int>(0xfc7ee074), 0x326ce92a,
0x366d13b5, 0x17aaa731, static_cast<int>(0xeb83a675), 0x7781cb32,
0x4ec7c92d, 0x7f187521, 0x2cf346b4, static_cast<int>(0xad13310f),
static_cast<int>(0xb89cff2b), 0x12164de1, static_cast<int>(0xa865168d), 0x32b56cdf
};

MLCG::MLCG(int seed1, int seed2)
{
    initialSeedOne = seed1;
    initialSeedTwo = seed2;
    reset();
}

void
MLCG::reset()
{
    int seed1 = initialSeedOne;
    int seed2 = initialSeedTwo;

    //
    //	Most people pick stupid seed numbers that do not have enough
    //	bits. In this case, if they pick a small seed number, we
    //	map that to a specific seed.
    //
    if (seed1 < 0) {
	seed1 = (seed1 + 2147483561);
	seed1 = (seed1 < 0) ? -seed1 : seed1;
    }

    if (seed2 < 0) {
	seed2 = (seed2 + 2147483561);
	seed2 = (seed2 < 0) ? -seed2 : seed2;
    }

    if (seed1 > -1 && seed1 < SEED_TABLE_SIZE) {
	seedOne = seedTable[seed1];
    } else {
	seedOne = seed1 ^ seedTable[seed1 & (SEED_TABLE_SIZE-1)];
    }

    if (seed2 > -1 && seed2 < SEED_TABLE_SIZE) {
	seedTwo = seedTable[seed2];
    } else {
	seedTwo = seed2 ^ seedTable[ seed2 & (SEED_TABLE_SIZE-1) ];
    }
    seedOne = (seedOne % 2147483561) + 1;
    seedTwo = (seedTwo % 2147483397) + 1;
}

unsigned int MLCG::asLong()
{
    int k = seedOne % 53668;

    seedOne = 40014 * (seedOne-k * 53668) - k * 12211;
    if (seedOne < 0) {
	seedOne += 2147483563;
    }

    k = seedTwo % 52774;
    seedTwo = 40692 * (seedTwo - k * 52774) - k * 3791;
    if (seedTwo < 0) {
	seedTwo += 2147483399;
    }

    int z = seedOne - seedTwo;
    if (z < 1) {
	z += 2147483562;
    }
    return( (unsigned long) z);
}
