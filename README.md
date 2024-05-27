## IonTrap-2QG

## Overview
IonTrap-2QG is a project focused on simulating and analyzing ion-trap quantum gates. The repository contains code and data for the implementation and testing of quantum gates using ion-trap technology.

## Repository Structure
- **data/**: Contains ion-trap parameters used for simulations and the configuration files.
- **include/**: Third-party libraries.
- **src/**: Source code for simulations and analyses.
- **.gitignore**: Specifies files to be ignored by git.

## Getting Started

### Prerequisites
- C++ compiler
- Python 3.x
- CMake

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/shakibv/iontrap-2qg.git
    ```
2. Navigate to the project directory:
    ```sh
    cd iontrap-2qg
    ```
3. Build the project:
    ```sh
    mkdir build
    cd build
    cmake ..
    make
    ```

## Usage
Run the simulations by executing the compiled binaries in the `build` directory. Refer to the specific examples in the `src` folder for guidance on running simulations and analyzing results.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
IonTrap-2QG adapts and expands the `QSD 1.3.5` library and uses the `cnpy` and `differential-evolution` libraries.

### QSD
`QSD` © 1996-2004 Todd Brun and Rüdiger Schack, is described in Comput.Phys.Commun. 102 (1997) 210-228.
<details>
<summary>Copyright (C) 1995  Todd Brun and Ruediger Schack</summary>

```text
Copyright (C) 1995  Todd Brun and Ruediger Schack

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

----------------------------------------------------------------------
If you improve the code or make additions to it, or if you have
comments or suggestions, please contact us:

Dr. Todd Brun			        Tel    +44 (0)171 775 3292
Department of Physics                      FAX    +44 (0)181 981 9465
Queen Mary and Westfield College           email  t.brun@qmw.ac.uk
Mile End Road, London E1 4NS, UK

Dr. Ruediger Schack                        Tel    +44 (0)1784 443097
Department of Mathematics                  FAX    +44 (0)1784 430766
Royal Holloway, University of London       email  r.schack@rhbnc.ac.uk
Egham, Surrey TW20 0EX, UK
```
</details>

### differential-evolution
`differential-evolution` © 2017 Adrian Mitchel, can be accessed from its [GitHub repository](https://github.com/adrianmichel/differential-evolution).
<details>
<summary>BSD 3-Clause License Copyright (c) 2017, Adrian Michel</summary>

```text
BSD 3-Clause License

Copyright (c) 2017, Adrian Michel
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
</details>

### cnpy
`cnpy` © 2011 Carl Rogers, can be accessed from its [GitHub repository](https://github.com/rogersce/cnpy/).
<details>
<summary>The MIT License Copyright (c) Carl Rogers, 2011</summary>

```text
The MIT License

Copyright (c) Carl Rogers, 2011

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
</details>

## Contact
For questions or issues, please open an issue in this repository.
