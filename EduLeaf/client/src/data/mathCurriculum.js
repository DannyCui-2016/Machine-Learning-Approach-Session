// Mock Curriculum Data for Educational Systems

export const CURRICULUM_DATA = {
  cambridge: {
    12: { // Year 12 AS Level
      title: "AS Level Mathematics",
      subtitle: "Cambridge International",
      description: "The primary half of the A-Level course. Covers Pure Mathematics 1 and either Mechanics or Probability & Statistics 1.",
      goals: [
        "Master advanced algebra, functions, and coordinate geometry.",
        "Understand the foundational theories of Calculus (Differentiation & Integration).",
        "Apply mathematical models to physics scenarios (Mechanics) or perform data analysis (Statistics)."
      ],
      topics: [
        {
          id: "pure-1",
          title: "Pure Mathematics 1",
          items: [
            {
              name: "1. Quadratics",
              concepts: ["Completing the square", "Discriminant", "Roots of quadratic equations", "Quadratic inequalities"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/quadratics/"
            },
            {
              name: "2. Functions",
              concepts: ["Domain & range", "Composite functions", "Inverse functions"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/functions/"
            },
            {
              name: "3. Coordinate Geometry",
              concepts: ["Equation of a straight line", "Parallel and perpendicular lines", "Length and mid-point of a line segment"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/coordinate-geometry/"
            },
            {
              name: "4. Circular Measure",
              concepts: ["Radian measure", "Arc length", "Area of a sector"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/circular-measure/"
            },
            {
              name: "5. Trigonometry",
              concepts: ["Exact values", "Trigonometric graphs", "Trigonometric identities and equations"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/trigonometry/"
            },
            {
              name: "6. Series",
              concepts: ["Binomial expansion", "Arithmetic progressions", "Geometric progressions"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/series/"
            },
            {
              name: "7. Differentiation",
              concepts: ["Derivative of x^n", "Chain rule", "Tangents and normals", "Stationary points"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/differentiation/"
            },
            {
              name: "8. Integration",
              concepts: ["Indefinite integration", "Definite integrals", "Area under a curve"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/integration/"
            }
          ]
        },
        {
          id: "stats-1",
          title: "Probability & Statistics 1",
          items: [
            {
              name: "1. Representation of Data",
              concepts: ["Stem-and-leaf diagrams", "Box-and-whisker plots", "Histograms", "Cumulative frequency graphs"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/statistics-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/statistics-1/representation-of-data/"
            },
            {
              name: "2. Permutations and Combinations",
              concepts: ["Factorial notation", "Arrangements", "Selections"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/statistics-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/statistics-1/permutations-and-combinations/"
            },
            {
              name: "3. Probability",
              concepts: ["Mutually exclusive events", "Independent events", "Conditional probability", "Tree diagrams"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/statistics-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/statistics-1/probability/"
            },
            {
              name: "4. Discrete Random Variables",
              concepts: ["Probability distributions", "Expectation", "Variance"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/statistics-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/statistics-1/discrete-random-variables/"
            },
            {
              name: "5. The Normal Distribution",
              concepts: ["Standard normal distribution", "Standardising a normal variable", "Normal approximation to the binomial distribution"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/statistics-1-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/statistics-1/the-normal-distribution/"
            }
          ]
        }
      ],
      pastPapers: [
        {
          year: "2023",
          season: "May/June",
          paperUrl: "https://pastpapers.papacambridge.com/viewer/cie/igcse-maths-0580-2023-may-june",
          markSchemeUrl: "https://pastpapers.papacambridge.com/viewer/cie/igcse-maths-0580-2023-may-june"
        },
        {
          year: "2022",
          season: "Oct/Nov",
          paperUrl: "https://pastpapers.papacambridge.com/viewer/cie/igcse-maths-0580-2022-oct-nov",
          markSchemeUrl: "https://pastpapers.papacambridge.com/viewer/cie/igcse-maths-0580-2022-oct-nov"
        }
      ]
    },
    13: { // Year 13 A2 Level
      title: "A2 Level Mathematics",
      subtitle: "Cambridge International",
      description: "The secondary half of the A-Level course. Covers Pure Mathematics 3 and either Mechanics or Probability & Statistics 2.",
      goals: [
        "Master complex algebra, logarithms, and advanced trigonometry.",
        "Deepen understanding of Calculus with advanced techniques (Integration by parts, Differential equations).",
        "Perform complex vector operations and hypothesis testing."
      ],
      topics: [
        {
          id: "pure-3",
          title: "Pure Mathematics 3",
          items: [
            {
              name: "1. Algebra",
              concepts: ["Modulus function", "Division of polynomials", "Partial fractions"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/algebra/"
            },
            {
              name: "2. Logarithmic and Exponential Functions",
              concepts: ["Logarithms to base 10 and base e", "Solving exponential equations"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/logarithmic-and-exponential-functions/"
            },
            {
              name: "3. Trigonometry",
              concepts: ["Secant, cosecant and cotangent", "Addition formulae", "Double angle formulae"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/trigonometry/"
            },
            {
              name: "4. Differentiation",
              concepts: ["Derivative of e^x, ln x, sin x, cos x, tan x", "Product and quotient rules", "Implicit differentiation"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/differentiation/"
            },
            {
              name: "5. Integration",
              concepts: ["Integration of e^x, 1/x, sin x, cos x", "Integration by substitution", "Integration by parts"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/integration/"
            },
            {
              name: "6. Numerical Solutions of Equations",
              concepts: ["Location of roots", "Iterative methods"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/numerical-solutions-of-equations/"
            },
            {
              name: "7. Vectors",
              concepts: ["Vector equation of a line", "Intersection of two lines", "Angle between two lines"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/vectors/"
            },
            {
              name: "8. Differential Equations",
              concepts: ["Formulating simple differential equations", "Solving by separating variables"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/differential-equations/"
            },
            {
              name: "9. Complex Numbers",
              concepts: ["Imaginary numbers", "Argand diagrams", "Solving equations with complex roots"],
              tutorialLink: "https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/",
              practiceLink: "https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/complex-numbers/"
            }
          ]
        }
      ],
      pastPapers: [
        {
          year: "2023",
          season: "May/June",
          paperUrl: "https://pastpapers.papacambridge.com/viewer/cie/igcse-maths-0580-2023-may-june",
          markSchemeUrl: "https://pastpapers.papacambridge.com/viewer/cie/igcse-maths-0580-2023-may-june"
        }
      ]
    }
  }
};

export const getCurriculumData = (system, year) => {
  if (CURRICULUM_DATA[system] && CURRICULUM_DATA[system][year]) {
    return CURRICULUM_DATA[system][year];
  }
  return null;
};
