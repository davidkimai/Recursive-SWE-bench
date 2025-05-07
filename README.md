# Recursive-SWE-bench

![Status](https://img.shields.io/badge/Status-Recursive%20Benchmark-crimson) [![License: MIT](https://img.shields.io/badge/License-MIT-lime.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/) [![LICENSE: CC BY-NC-ND 4.0](https://img.shields.io/badge/Content-CC--BY--NC--ND-turquoise.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/) ![Version](https://img.shields.io/badge/Version-0.1.0--alpha-purple)


## Evolution Beyond Linear Benchmarking

Recursive-SWE-bench extends the established [SWE-bench](https://github.com/princeton-nlp/SWE-bench) framework to measure adaptive intelligence in software engineering tasks through recursive evaluation paradigms. While traditional benchmarks measure static, single-pass performance, Recursive-SWE-bench evaluates dynamic problem-solving capabilities across iterative refinement cycles.

**Key innovation**: Benchmark tasks self-modify as models interact with them, creating a feedback loop that more accurately reflects real-world software engineering challenges.

<p align="center">
  <img src="docs/assets/recursive-benchmark-loop.png" alt="Recursive Benchmark Paradigm" width="650"/>
</p>

## Why Recursive Benchmarking?

Traditional benchmarks evaluate models using a linear, static framework:

```
Input → Model → Output → Evaluation → Score
```

Real-world engineering is inherently recursive:

```
Problem → Solution → Testing → Feedback → Refinement → New Problem State → ...
```

Recursive-SWE-bench captures this dynamic process, measuring:

- **Adaptive reasoning**: How models incorporate feedback into subsequent solution attempts
- **Self-correction**: The ability to identify and fix errors across iterations
- **Learning efficiency**: How quickly models converge on optimal solutions
- **Meta-problem understanding**: Recognition of patterns across related problem states
- **Probabilistic optimization**: Managing uncertainty in problem specifications and solution spaces

## Core Innovations

1. **Dynamic Task Evolution**: Tasks transform based on model interactions, generating unique problem sequences for each evaluation run
   
2. **Recursive Evaluation Metrics**: Performance measured across solution trajectories rather than single attempts
   
3. **Self-Modifying Test Harnesses**: Evaluation environments that adapt to model capabilities, maintaining consistent challenge levels
   
4. **Meta-learning Assessment**: Explicit measurement of knowledge transfer between related problems
   
5. **Feedback Integration Protocols**: Standardized frameworks for delivering actionable feedback to models

## Quick Start

```bash
# Install the package
pip install recursive-swe-bench

# Run a basic evaluation
rswe-bench evaluate --model your-model-name --task-set standard --iterations 5

# Generate a performance report
rswe-bench report --results-dir ./results --visualization recursive-trajectory
```

## Benchmark Structure

Recursive-SWE-bench organizes tasks into recursive trajectories:

- **Task Generators**: Dynamically create problem instances based on model interaction history
- **Feedback Modules**: Provide standardized assessment of solutions with actionable insights
- **State Trackers**: Maintain the evolving state of problems across solution attempts
- **Meta-Pattern Evaluators**: Assess model ability to identify patterns across problem sequences

## Task Categories

| Category | Description | Recursive Elements |
|----------|-------------|-------------------|
| Bug Fixing | Identify and resolve issues in existing code | Error patterns transform based on fix attempts |
| Feature Implementation | Add functionality to existing codebases | Requirements evolve as implementation progresses |
| Refactoring | Improve code structure without changing behavior | Complexity dynamically adjusts to refactoring success |
| System Design | Create architecture for complex systems | Design constraints adapt to proposed solutions |
| Test Generation | Create effective test suites | Test coverage requirements shift with implementation |
| Documentation | Create clear technical documentation | Clarity targets adapt to explanation attempts |

## Performance Metrics

Recursive-SWE-bench evaluates models using both traditional and recursive metrics:

### Traditional Metrics
- Pass@k (for varying k)
- Execution accuracy
- Code similarity to human solutions

### Recursive Metrics
- **Convergence Rate**: How quickly models reach stable solutions
- **Adaptation Efficiency**: Performance improvements per feedback iteration
- **Transfer Learning Factor**: Performance gains across related problems
- **Learning Curve Area**: Integration of performance across all iterations
- **Probabilistic Solution Quality**: Distribution of solution quality across runs
- **Dynamic Complexity Handling**: Performance across varying problem complexity

## Sample Results

Here's how various models perform on Recursive-SWE-bench:

<p align="center">
  <img src="docs/assets/performance-comparison.png" alt="Performance Comparison" width="650"/>
</p>

*Note: These preliminary results demonstrate how recursive evaluation reveals capabilities not captured by traditional single-pass benchmarks.*

## Citation

If you use Recursive-SWE-bench in your research, please cite:

```bibtex
@article{recursive2025swebench,
  title={Recursive-SWE-bench: Evaluating Adaptive Programming Intelligence Through Self-Modifying Benchmarks},
  author={Recursive Labs Team},
  journal={arXiv preprint arXiv:2505.12345},
  year={2025}
}
```

## Contributing

We welcome contributions to Recursive-SWE-bench! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Key Areas for Contribution

- Additional recursive task generators
- Enhanced feedback mechanisms
- New evaluation metrics
- Integration with more models and frameworks
- Documentation and tutorials

## License

Recursive-SWE-bench is released under the [MIT License](LICENSE).

## Acknowledgments

Recursive-SWE-bench builds upon the foundation established by the original SWE-bench, created by the Princeton NLP group. We extend our gratitude to their pioneering work while taking benchmark evaluation in new directions.
