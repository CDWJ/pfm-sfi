# Solid-Fluid Interaction on Particle Flow Maps

[**Duowen Chen** ](https://cdwj.github.io/), [Zhiqi Li](https://zhiqili-cg.github.io/), [Junwei Zhou](https://zjw49246.github.io/website/), [Fan Feng](https://sking8.github.io/), [Tao Du](https://people.iiis.tsinghua.edu.cn/~taodu/), [Bo Zhu](https://www.cs.dartmouth.edu/~bozhu/)

[![webpage](https://img.shields.io/badge/Project-Homepage-green)](https://cdwj.github.io/projects/pfm-sfi-project-page/index.html)
[![paper](https://img.shields.io/badge/Paper-Preprint-red)](https://cdwj.github.io/projects/pfm-sfi-project-page/static/pdfs/SASIA_2024__Solid_Fluid_Interaction_on_Particle_Flow_Maps.pdf)
[![code](https://img.shields.io/badge/Source_Code-Github-blue)](https://github.com/CDWJ/pfm-sfi)

This repo stores the source code of our SIGGRAPH ASIA 2024 paper **Solid-Fluid Interaction on Particle Flow Maps**

<figure>
  <img src="./representative.jpg" align="left" width="100%" style="margin: 0% 5% 2.5% 0%">
  <figcaption> We demonstrate our method’s efficacy with various examples of fluid-solid interaction, including a swimming fish, long silk flags, a Koinobori, and a
 falling parachute along with its trajectory, all exhibiting strong vortex dynamics and solid-vortex interactions.</figcaption>
</figure>
<br />

## Abstract

We propose a novel solid-fluid interaction method for coupling elastic solids with impulse flow maps. Our key idea is to unify the representation of fluid and solid components as particle flow maps with different lengths and dynamics. The solid-fluid coupling is enabled by implementing two novel mechanisms: first, we developed an impulse-to-velocity transfer mechanism to unify the exchanged physical quantities; second, we devised a particle path integral mechanism to accumulate coupling forces along each flow-map trajectory. Our framework integrates these two mechanisms into an Eulerian-Lagrangian impulse fluid simulator to accommodate traditional coupling models, exemplified by the Material Point Method (MPM) and Immersed Boundary Method (IBM), within a particle flow map framework. We demonstrate our method's efficacy by simulating solid-fluid interactions exhibiting strong vortical dynamics, including various vortex shedding and interaction examples across swimming, falling, breezing, and combustion.

## Usage

```bash
python run.py
```