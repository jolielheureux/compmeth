import gaussxw
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sympy as sp


#functions that my funciton will integrate 

#main/default function is f(t) = e^(-t^2) but want to add other options to script

def func(t):
    """E(t) = e^(-t^2)""" #default function that will be integrated
    y = np.e**(-t**2)
    return y

#other common options for functions 
def sine(t):
    """sin(t)"""
    return np.sin(t)

def cosine(t):
    """cos(t)"""
    return np.cos(t)


def cubic(t):
    """t^3"""
    return t**3

def linear(t):
    """t"""
    return t

def exp_decay(t):
    """e^(-t)"""
    return np.e**(-t)

def abs_val(t):
    """|t|"""
    return np.abs(t)

def squared(t):
    """t^2"""
    return t**2
def square_root(t):
    """t^(1/2)"""
    return t**(.5)

#turn functions into a library to choose from 

FUNCTIONS = {
    "func": func,
    "sine": sine,
    "cosine": cosine,
    "cubic": cubic,
    "linear": linear,
    "exp_decay": exp_decay,
    "abs_val": abs_val,
    "squared": squared,
    "square_root": square_root,
}

#if someone wants to add their own random funciton in terms of t they can type it as a string 
def make_function_from_string(expr_str, var_name='t'):
    """
    Takes a math expression as a string (e.g. '2**t', 'sin(t)*t')
    and returns a callable function f(t) that works with numpy arrays.
    """
    t = sp.symbols(var_name)
    expr = sp.sympify(expr_str)          # parse the string 
    f = sp.lambdify(t, expr, modules=['numpy'])  # turn it into a function f
    f.__doc__ = expr_str                 # sets docstring-based of newly created function f
    return f

#my funciton will take a function, a value, b value, and step size as the arguments
#default will be set as the values given in the HW problem 
#The value will be given for each E(x) from 0 to x whith x ranging from 0 to 3 in step sizes of .1
#E(x) is the integral of 0 to x of e^(-t^2) 
# you can change the bounds 
#I also wanted to plot the f(t) funciton, shade the FINAL region we integrate! 

def main():
    parser = argparse.ArgumentParser(description= "Numerically compute E(x) = the integral from 0 to x of a function f(t), "
                     "using Gaussian quadrature, for x ranging from --a to --b in steps of --step_size. "
                     "Prints E(x) at each step, then plots the selected function f(t) with a shaded "
                     "region showing one integral over [a, b], and a separate plot of E(x) vs x. "
                     "Choose a function from the built-in library, or supply your own expression in t. "
                     "Default function is f(t) = e^(-t^2).")
    parser.add_argument("--function", type=str, default="func", help = "Choose a function from the library (sine, cosine, cubic, linear, exp_decay, abs_val, squared, square_root)"
                        "or a custom expression in t, e.g. '2+t' or 'sin(t)*t'  Wrap custom expressions in quotes. Default function is E(t) = e^{-t^2}")
    parser.add_argument("--a", type=float, default= 0, help = "Lower bound of integral. Default is a = 0")
    parser.add_argument("--b", type=float, default= 3, help = "Upper bound of integral. Default is b = 3")
    parser.add_argument("--step_size", type=float, default= .1, help = "Step size. Default is step_size = 0.1")
    args = parser.parse_args()

    a = args.a
    b = args.b
    step_size = args.step_size

    #for some step sizes it breaks so want to make sure that the minimum people will input is .05
    if step_size < 0.05:
        print(f"NOTE: step_size {step_size} is too small; using minimum of 0.05 instead.")
        step_size = 0.05
    
    #if funciton is from my list pull the function 
    if args.function in FUNCTIONS:
        function_choice = FUNCTIONS[args.function]
    #if its not then make a custom function 
    else:
        function_choice = make_function_from_string(args.function)

    #use Gaussian Quadrature to integrate
    
    #Part (a) of the homework
    E_values = []
    x_steps = np.arange(a, b + step_size / 2, step_size) 
    print(f"E(x) = integral of '{args.function}' from {a} to x, in steps of {step_size}:")
    for i in x_steps:
        if i == a:
            E_x = 0.0
        else:
            n_points = max(len(np.arange(a, i, step_size)), 1) #step sizes increase as value of t gets larger
            xg, wg = gaussxw.gaussxw(n_points, a, i)
            E_x = np.sum(wg * function_choice(xg))
        E_values.append(E_x)  # add this line
        print(f"  E({i:.2f}) = {E_x:.6f}")


#full integral
    list = np.arange(a,b,step_size)
    x,w = gaussxw.gaussxw(len(list),a,b)
    summation = np.sum(w*function_choice(x))
    print(f"Integral of '{args.function}' from t={a} to t={b}: {summation:.4f}")
    
#set boundaries of plot so that it can include x-values but will default to -4 and 4 so that we can see E(t)
    if a >= -4:
        t1 = -4
    else:
        t1 = a
    if b <= 4:
        t2 = 4
    else:
        t2 = b

    t = np.linspace(t1, t2, 100) #x axis 

    fig, ax = plt.subplots(figsize=(7, 5))
    
    if args.function == parser.get_default("function"): #nice labels for default function
        ax.plot(t, function_choice(t), color='#1f77b4', linewidth=2, label=r'$f(t) = e^{-t^2}$') #label original function
    else: #no label if user wants to input another funciton
        ax.plot(t, function_choice(t), color='#1f77b4', linewidth=2)


    segment = np.linspace(a,b,1000)

    #fill where user wants to do integration
    plt.fill_between(x, function_choice(x), alpha=0.3, color='#1f77b4',
                        label=f'∫ from {a} to {b} ≈ {summation:.4f}') #fill area of integration 

    #plot the segment that we are integrating in red 
    plt.plot(segment, function_choice(segment), color='red')

    #plot the bounds of integral as dashed vertical lines to clearly see
    ax.axvline(a, color='gray', linestyle='--', linewidth=1)
    ax.axvline(b, color='gray', linestyle='--', linewidth=1)
    ax.axhline(0, color='black', linewidth=0.8)

    #label x-axis t
    ax.set_xlabel('t')

    #label y-axis E(t)
    if args.function == parser.get_default("function"):
        ax.set_ylabel(r'$f(t) = e^{-t^2}$')
    else:
        ax.set_ylabel('E(t)')
    ax.set_title('Numerical Integration by Gaussian Quadrature')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
 

#Part (b) of the Homework
# E(x) as a function of x
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(x_steps, E_values, color='#1f77b4', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('x')
    ax2.set_ylabel('E(x)')
    ax2.set_title('E(x) vs x')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    