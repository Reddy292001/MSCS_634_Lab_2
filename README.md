# Wine Classification with KNN and Radius Neighbors (RNN)

This repo contains a Jupyter Notebook that compares two **distance-based** classifiers on the **scikit-learn Wine dataset**:

- **K-Nearest Neighbors (KNN)**
- **Radius Neighbors (RNN)** *(RNN here means **Radius Neighbors**, not Recurrent Neural Networks)*

---

## Goal of the Lab

The main goals were to:

- Build and evaluate **KNN** and **Radius Neighbors** classifiers.
- Study how key parameters affect accuracy:
  - **k** (number of neighbors) for KNN
  - **radius** (distance threshold) for Radius Neighbors
- Plot accuracy trends to find good parameter values.
- Compare both models and explain when each one makes sense.

---

## Dataset

- **Dataset:** Wine (from `sklearn.datasets`)
- **Task:** Multi-class classification (predict the wine class based on features)

---

## What We Did

### 1) Data Preprocessing (Important!)
Since both models rely on distances, feature scaling matters a lot.

We used **StandardScaler** to:
- set feature mean to **0**
- set feature standard deviation to **1**

This prevents features with larger ranges from dominating the distance calculations.

---

### 2) KNN Model (K-Nearest Neighbors)

We tested several values of **k** and measured test accuracy.

**Key behavior we observed:**
- **Low k (e.g., 1):** often overfits (too sensitive to noise)
- **Middle k (e.g., 5, 11):** best balance → highest accuracy (sweet spot)
- **High k (e.g., 21):** can underfit (too “smoothed out” decisions)

---

### 3) Radius Neighbors Model (RNN = Radius Neighbors)

We tested several **radius** values and measured accuracy.

**Key behavior we observed:**
- **Too small radius:** a test point may have **zero neighbors**, which can raise a `ValueError`
- **Medium radius:** accuracy improves at first as more neighbors are included
- **Too large radius:** includes points from many classes → accuracy drops (class boundaries blur)

To keep the loop running, we used a **try/except** block:
- if no neighbors were found, we assigned accuracy = **0** for that radius value

---

## Key Insights (Summary)

### KNN (Overall Better on Wine)
- KNN was **more stable** and usually **more accurate** on this dataset.
- The Wine dataset has fairly consistent density and clear clusters, so using a **fixed number of neighbors** worked well.

### Radius Neighbors (More Sensitive)
- Choosing one global radius is tricky:
  - too small → no neighbors
  - too large → mixed classes
- This made Radius Neighbors **less reliable** for the Wine dataset.

---

## Conclusion

- **KNN is the better and simpler choice** for datasets like Wine (clean clusters, fairly uniform density).
- **Radius Neighbors can be useful** when the dataset has big density differences and a distance cutoff makes more sense than a fixed neighbor count.

---

## Challenges We Faced

- **Picking a good radius:** not intuitive in multi-dimensional space
- **Handling errors:** very small radii can cause no-neighbor cases
- **Scaling features:** without standardization, distance-based models can behave poorly

---

## How to Run

### 1) Install Dependencies
```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

### 2) Open the Notebook
Run one of the following:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

### 3) Execute the Notebook
Open and run:
- `Wine_KNN_RNN_Lab.ipynb`

---

## Repository Contents

- `Wine_KNN_RNN_Lab.ipynb` — main notebook with experiments and plots
- `README.md` — this file

---
## Author
Sai Venkata Bharath Reddy Singareddy - MSCS 634
