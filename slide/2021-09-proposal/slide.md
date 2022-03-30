---
marp: true
theme: default
paginate: true
color: blackz
footer: "Zhenyu Wei, Sep 2021"
header: <img src="./images/seu_logo.png" height=100>
---
<style>
  section {
      background-color: white;
      font-family: 'Helvetica',  !important;
      font-size: 35px;
    }
  
  header {
    position: absolute;
    left: 1146px;
    top: 550px;
  }

  section::after {
    font-weight: bold;
    text-shadow: 1px 1px 0 #fff;
    content: attr(data-marpit-pagination) ' / ' attr(data-marpit-pagination-total);
    position: absolute;
    left: 1170px;
    top: 660px;
  }

  section h1{
    font-size: 55px;
  }
  section h2{
    font-size: 60px;
  }
  section h3{
    font-size: 45px
  }
  section h4{
    font-size: 35px
  }
  img[alt~='center']{
    display: block;
    margin-left: auto;
    margin-right: auto;
  }
  .katex {
    font-size: 35px;
  }
  section.split {
    overflow: visible;
    display: grid;
    grid-template-columns: 600px 550px;
    grid-template-rows: 100px auto;
    grid-template-areas: "leftpanel rightpanel";
}
/* debug */
section.split h3, 
section.split .ldiv, 
section.split .rdiv { border: 0pt dashed dimgray; }
section.split h3 {
    grid-area: slideheading;
    font-size: 50px;
}
section.split .ldiv { grid-area: leftpanel; }
section.split .rdiv { grid-area: rightpanel; }
  
</style>
<center>

# A differentiable representation of 
# solvent-solute interface

<br>

Reporter: **Zhenyu Wei** 

Mentor: **Yunfei Chen**

School of mechanical engineering
Southeast University


---

## <center> Inspiration </center>

---

### Implicit solvent model


![w:1000px center](./images/multiscale-modeling.png)

---

### Poisson-Boltzmann Equation

<br>

$$
\nabla\varepsilon(\mathbf{r})\nabla\phi(\mathbf{r}) = - \rho_{mol}(\mathbf{r}) -\lambda(\mathbf{r})\kappa^2\mathrm{sinh}\left(-\frac{z_+e\phi(\mathbf{r})}{kT}\right) 
$$

<br>

A <b><font color=red size=7>differentiable</font></b>  expression of $\varepsilon_r$ is vital for a precise solution of **Poisson-Boltzmann Equation** (PBE)

---

<!-- _class: split -->
<div class=rdiv>

![w:450px center](./images/1a1n.png)

</div>

<div class=ldiv>
<font size=6>

&ensp;<b><font size=9> PBE for globular protein </font></b>
<br>

- The surface can be interpreted as a **complex spatial distribution**
  
- The spatial distribution is highly **non-linear**

- The distribution depends on all of the atom's **position** and **type**

</font>


&ensp;&ensp;&ensp;&ensp;<font size=3> PDB id of protein shown right: 1A1N</font>
</div>

---

### Current solution: Van der Waals surface

![w:800px center](./images/vdw_surface.png)

---

### Current obstacles

- Representation is **not smooth**

- Representation is **not expressed explicitly**

- Representation based on **Hard sphere model**
    - Highly approximated
    - Hyper parameter required

### Target

A smooth, differentiable, interface representation for solution of **PBE**

---

## <center> Method </center>

---

### Basic Idea

- Using deep neural network to represent the non-linear interface

- Deep neural network are naturally differentiable

### Obstacle

- Handling length-varied input: the positions and types of protein's atom

---

### Architecture

![w:900px center](./images/TSSIR_str.png)

---

### Dataset

<br>

- Downloaded **12748** structures from PDB website

- Patched and solvated **10962** structures in $60\times60\times60\ A$ box

- Currently, sampled **4458** structures

  - Label solution atom as 1
  
  - Label solvent atom as 0


---

## <center> Result and discussion </center>

---

<!-- _class: split -->
<div class=rdiv>

![w:550px center](./images/1iqg_surface.png)

</div>

<div class=ldiv>
<font size=6>
<br>

&ensp;<b><font size=9> Feasibility test </font></b>


- Train network on tiny dataset

  - 28 protein structures

  - Each with 25 samples

- Average accuracy: 92%  

- Isosurface with output value of TSSIR between [0.495, 0.505]

</font>


&ensp;&ensp;&ensp;&ensp;<font size=3> PDB id of protein shown right: 1IQG</font>
</div>

---

## <center> Thanks for your attention </center>
## <center> Q & A </center>