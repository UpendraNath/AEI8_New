# Certification Challenge Task List

## Submit Here (Due October 21 by 7:00 PM ET)

[Certification Challenge](https://forms.gle/4viHEd5BgAwW7mbi7)

## Overview

Welcome to the middle of the course!  We’re 5 weeks in and we’ve covered a lot of ground:

- reflect on the general skills you’ve learned and apply them to a new use case that interests you... or potential investors
- this is what the Certification challenge is all about - putting **your skills to the test**!

## Introduction

All good stories, like all production AI applications, rely on setting the right context.

What’s the context for your **CERTIFICATION CHALLENGE submission**?  That’s the first thing you need to figure out!

REMEMBER:  this is not directly tied to **Demo Day**, and can focus on other domains than your **Demo Day application**...

### 😎 AI Product Management

> AI product management asks “**What** should I build, ship, and share, and why?”

There are three primary questions **you need to answer** *before you can begin* the certification challenge:

1. What **problem** are you trying to solve?  
*Why is this a problem?*
2. What is your proposed **solution**?  
*Why is this the best solution?*
3. Who is the **audience** that has this problem and would use your solution?  
*Do they nod their head up and down when you talk to them about it?*

> Problem, Solution, Audience.

### 🧑‍💻 AI Engineering

## Task 1: Defining your Problem and Audience

**You are an AI Solutions Engineer**.

**What** problem do you want to solve?  **Who** is it a problem for?

> Articulate the problem and the user of your application

*Hint:*

- *Create a list of potential questions that your user is likely to ask!*
- *What is the user’s job title, and what is the part of their job function that you’re trying to automate?*

### Deliverables

1. Write a succinct 1-sentence description of the problem
2. Write 1-2 paragraphs on why this is a problem for your specific user

## Task 2: Propose a Solution

Now that you’ve defined a problem and a user, *there are many possible solutions*.  Choose one, and articulate it.

> Articulate your proposed solution

*Hint:*  

- *Paint a picture of the “better world” that your user will live in.  How will they save time, make money, or produce higher-quality output?*
- *Recall the [LLM Application stack](https://a16z.com/emerging-architectures-for-llm-applications/) we’ve discussed at length*

### Deliverables

1. Write 1-2 paragraphs on your proposed solution.  How will it look and feel to the user?
2. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice.
    1. LLM
    2. Embedding Model
    3. Orchestration
    4. Vector Database
    5. Monitoring
    6. Evaluation
    7. User Interface
    8. (Optional) Serving & Inference
3. Where will you use an agent or agents?  What will you use “agentic reasoning” for in your app?

## Task 3: Dealing with the Data

**You are an AI Systems Engineer.**  The AI Solutions Engineer has handed off the plan to you.  Now *you must identify some source data* that you can use for your application.  

Assume that you’ll be doing at least RAG (e.g., a PDF) with a general agentic search (e.g., a search API like [Tavily](https://tavily.com/) or [SERP](https://serpapi.com/)).

> Collect data for (at least) RAG and choose (at least) one external API

*Hint:*

- *Ask other real people (ideally the people you’re building for!) what they think.*
- *What are the specific questions that your user is likely to ask of your application?  **Write these down**.*

### Deliverables

1. Describe all of your data sources and external APIs, and describe what you’ll use them for.
2. Describe the default chunking strategy that you will use.  Why did you make this decision?
3. [Optional] Will you need specific data for any other part of your application?   If so, explain.

## Task 4: Building an End-to-End Agentic RAG Prototype

> Build an end-to-end Agentic RAG application using a production-grade stack and your choice of commercial off-the-shelf model(s)

### Deliverables

1. Build an end-to-end prototype and deploy it to a *local* endpoint
2. (Optional) use locally-hosted OSS models instead of LLMs through the OpenAI API

## Task 5: Creating a Golden Test Data Set

**You are an AI Evaluation & Performance Engineer.**  The AI Systems Engineer who built the initial RAG system has asked for your help and expertise in creating a "Golden Data Set" for evaluation.

> Prepare a test data set (either by generating synthetic data or by assembling an existing dataset) to baseline an initial evaluation with RAGAS.

### Deliverables

1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevancy, context precision, and context recall.  Provide a table of your output results.
2. What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

## Task 6: The Benefits of Advanced Retrieval

**You are an AI Systems Engineer.**  The AI Evaluation and Performance Engineer has asked for your help in making stepwise improvements to the application. They heard that “as goes retrieval, so goes generation” and have asked for your expertise.

> Install an advanced retriever of your choosing in our Agentic RAG application. 

### Deliverables

1. Describe the retrieval techniques that you plan to try and to assess in your application.  Write one sentence on why you believe each technique will be useful for your use case.
2. Test a host of advanced retrieval techniques on your application.

## Task 7: Assessing Performance

**You are the AI Evaluation & Performance Engineer**.  It's time to assess all options for this product.

> Assess the performance of the naive agentic RAG application versus the applications with advanced retrieval tooling

### Deliverables

1. How does the performance compare to your original RAG application?  Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements.  Provide results in a table.
2. Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?

## Your Final Submission

### Final Deliverable

Please include the following in your final submission:

1. A public (or otherwise shared) link to a **GitHub repo** that contains:
    1. A 5-minute (OR LESS) loom video of a live **demo of your application** that also describes the use case.
    2. A **written document** addressing each deliverable and answering each question
    3. All relevant code
