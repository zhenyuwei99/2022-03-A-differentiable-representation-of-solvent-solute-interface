---
marp: true
theme: default
paginate: true
color: blackz
footer: "Zhenyu Wei, Sep 2021"
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


# A differentiable representation of solvent-solute interface

Reporter: **Zhenyu Wei**

Mentor: **Yunfei Chen**

<br>

**School of mechanical engineering
Southeast University**

---

## <center> Inspiration </center>

---