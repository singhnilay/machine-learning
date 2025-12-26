# Linear Regression From Scratch Using OOP (Python)

## Introduction

In this project, I implemented **Linear Regression from scratch using Python**, without using libraries like scikit-learn or TensorFlow.  
The main goal of this project was **not just to make linear regression work**, but to understand **how machine learning algorithms are actually built internally** using **Object-Oriented Programming (OOP)**.

This project helped me move from *using ML libraries* to *understanding ML systems*.

---

## What I Built

I built a complete **linear regression training pipeline** using separate classes, where each class has a clear responsibility.

The system consists of:

- A **Model** class to store parameters and make predictions
- A **Cost Function** class to measure error
- A **Gradient Calculator** to compute gradients
- An **Optimizer** to update parameters
- A **Trainer** to coordinate the training process

Each part is written in a separate file, similar to how real-world ML frameworks are structured.

---

## What I Learned

### 1. Linear Regression Is More Than a Formula

Earlier, I thought linear regression was just:

y = wx + b


But through this project, I learned that linear regression is actually a **system** that includes:

- Forward pass (prediction)
- Loss calculation
- Gradient computation
- Parameter updates
- Training loops

Understanding this made machine learning feel much less like magic.

---

### 2. Importance of Object-Oriented Programming in ML

By using OOP, I learned:

- How to **separate responsibilities** properly
- Why models should store parameters but not training logic
- Why cost functions and gradients should be stateless
- How optimizers modify models without knowing how gradients are computed

This design made the code:
- Easier to debug
- Easier to extend
- Much closer to industry-level ML code

---

### 3. How Gradient Descent Actually Works

Instead of memorizing formulas, I learned:

- What gradients represent
- Why shapes of matrices matter
- How each weight has its own gradient
- Why averaging over samples is important

Fixing bugs related to shapes helped me deeply understand the math behind training.

---

### 4. How ML Libraries Are Designed Internally

After building this project, I now understand how libraries like **scikit-learn** or **PyTorch** work internally:

- Models only store parameters
- Optimizers update parameters
- Trainers coordinate everything
- Components are loosely coupled

This project gave me confidence that I can **read and understand real ML source code**.

---

## How This Applies to Industry Standards

### 1. Clean Architecture

In industry, ML code must be:
- Maintainable
- Testable
- Scalable

This project follows:
- Single Responsibility Principle
- Modular design
- Clear data flow

These are the same principles used in production ML systems.

---

### 2. Extensibility

Because of this design, it is easy to:
- Add new cost functions
- Add new optimizers like Adam
- Add regularization
- Extend to multiple features

This is exactly how real ML teams evolve models over time.

---

### 3. Debugging and Reliability

In real companies:
- Silent bugs are dangerous
- Shape mismatches can break training

By building everything from scratch, I learned how to:
- Debug numerical issues
- Understand error messages
- Trust my implementation

This skill is very important for real-world ML engineering.

---

## Conclusion

This project helped me move from **using machine learning** to **understanding machine learning**.

I now feel confident about:
- How linear regression works internally
- How ML systems are structured
- How OOP is used in real ML pipelines

Most importantly, this project taught me that **good ML is not about fancy algorithms**, but about **clean design, correct math, and disciplined engineering**.

---

## Future Improvements

In the future, I plan to:
- Add regularization (Ridge / Lasso)
- Implement advanced optimizers like Adam
- Add plotting for loss curves
- Write unit tests
- Convert this into a small reusable ML library

This project has given me a strong foundation to build more advanced machine learning systems.
