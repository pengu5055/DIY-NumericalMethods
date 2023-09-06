# **Marko's makeshift PDE and ODE system solver package**

Here's a collection of methods that you can use in your work to solve PDEs and ODEs. The methods are not optimized for speed, but rather for ease of use and understanding. The Classes presented
are merely meant as a starting point for your own work, as some have the characteristics of whatever
equation they solve hard-coded into them and need a few modifications. For example in the FDMSolver
class, you're probably going to need to change the matrix of the system, to match your condition
and equation, as currently it is hard-coded to solve Schrodingers equation. The same goes for the
ColocationSolver class, where with a few modifications you can transform your equation into the correct form for the solver to work.

## **Installation**
Just like.. import my code into your project, and you're good to go. I'm not going to make a pip package for this, as it's not really meant for public use, but rather for my own use and for my friends. Requirements can be installed with
```bash
pip install -r requirements.txt
```

and I recommend you import the indivdual classes you need as such
```python
from FDMSolver import FDMSolver
```

## **Usage**
I provide some demonstration scripts in the examples folder, but here's a quick rundown of how to use the classes. Generally everything works the same in principle, where you define the starting parameters and grid size and such and supply those to the solver class at initialization. Then 
during work you do not need to reinput many of these values as they are already inherently part of the object. Solutions are obtained via solve() methods. Have a quick glance at the code for more information as there are docstrings and comments everywhere. For the purpose of keeping things
neat and tidy, I removed any built-in plotting capabilites from the classes but these can be easily added back in if needed. 


Much love, Marko (pengu5055) <3