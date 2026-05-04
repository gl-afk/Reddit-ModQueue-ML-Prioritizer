# Reddit-ModQueue-ML-Prioritizer

## About
This repository contains the source code for an academic proof-of-concept developed for a University thesis. The goal of the project is to enhance the efficiency of Reddit moderators by providing a ML-driven re-ranking of the ModQueue, based on community signals.

## Architecture
The project is divided into 2 main components:
1. For the frontend, a custom Chrome Extension that injects a sorting interface directly into the Reddit ModQueue DOM.
2. For the backend, a local Python server (`FastAPI`) responsible for fetching historical ModQueue data via the Reddit API, analyzing community signals, and evaluating a priority score using `scikit-learn` models.

**The project is currently in the _very_ initial setup and Exploratory Data Analysis (EDA) phase. The backend infrastructure and Chrome Extension UI are under active development in a local environment**.
