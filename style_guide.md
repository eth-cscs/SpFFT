# Style Guide for SpFFT
## Formatting
The formatting style is based on the google style guide with the following exceptions:
- Column size is limited to 100 instead of 80
- Access modifiers such as public and private are offset by -2

Clang-Format is used to format all files.

## Naming
The following rules are not strict and consistency when using external types is preferred.

### Files
Use underscores for separation.
File suffix:
- C++: .cpp and .hpp
- C: .c and .h
- CUDA: .cu and .cuh

Example
`my_class.cpp`

### Types
Use camelcase and start with a capital letter.
Example
`using MyType = int;`

### Variables
Use camelcase and start with lower case
Example:
`int myValue = 0;`

#### Class / Struct Members
Use a trailing underscore for non-public member variables. Public members are mamed like normal variables.

#### Functions
Function names use underscores and are all lower case
Example:
`my_function(int);`

#### namespaces
Namepsace are all lower case and use underscores.
Example:
` namespace my_space {}`

#### Macros
Macros are all capital and use underscores.
Example:
`#define MY_MACRO_VALUE 1`

