// Cambridge A-Level Topic-Based Learning Path
// Organised from foundations → advanced, fully covering AS + A2 syllabuses

export const CAMBRIDGE_TOPIC_PATH = [
  {
    stageId: 'foundations',
    stageTitle: 'Stage 1 – Foundations',
    stageDesc: 'Essential algebra and number skills every A-Level student must master first.',
    color: '#64B5F6', // light blue
    topics: [
      {
        id: 'indices-surds',
        title: 'Indices & Surds',
        difficulty: 1,
        concepts: [
          'Laws of indices (multiplication, division, power rules)',
          'Negative and fractional indices',
          'Simplifying surds (√)',
          'Rationalising the denominator',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/',
        practiceProblems: [
          {
            q: 'Simplify: 2⁵ × 2⁻³',
            a: '2⁵ × 2⁻³ = 2^(5−3) = 2² = **4**',
          },
          {
            q: 'Evaluate: (27)^(2/3)',
            a: '(27)^(2/3) = (∛27)² = 3² = **9**',
          },
          {
            q: 'Rationalise the denominator: 1 / (3 − √5)',
            a: 'Multiply top and bottom by (3 + √5):\n= (3 + √5) / [(3)² − (√5)²]\n= (3 + √5) / (9 − 5)\n= **(3 + √5) / 4**',
          },
        ],
      },
      {
        id: 'quadratics',
        title: 'Quadratics',
        difficulty: 1,
        concepts: [
          'Factorising quadratics',
          'Completing the square',
          'Quadratic formula',
          'Discriminant (Δ = b²−4ac) and nature of roots',
          'Quadratic inequalities',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/quadratics/',
        practiceProblems: [
          {
            q: 'Solve: x² − 5x + 6 = 0',
            a: 'Factorise: (x − 2)(x − 3) = 0\n**x = 2 or x = 3**',
          },
          {
            q: 'Complete the square: x² + 8x + 3',
            a: 'x² + 8x + 3 = (x + 4)² − 16 + 3 = **(x + 4)² − 13**',
          },
          {
            q: 'Find the number of real solutions of 3x² − 4x + 2 = 0.',
            a: 'Δ = (−4)² − 4(3)(2) = 16 − 24 = −8 < 0\n**No real solutions** (roots are complex).',
          },
        ],
      },
      {
        id: 'polynomials',
        title: 'Polynomials & Factor Theorem',
        difficulty: 2,
        concepts: [
          'Polynomial long division',
          'Factor theorem: if f(a) = 0 then (x − a) is a factor',
          'Remainder theorem',
          'Partial fractions (proper fractions)',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/algebra/',
        practiceProblems: [
          {
            q: 'Show that (x − 2) is a factor of f(x) = x³ − 3x² + 4.',
            a: 'f(2) = 8 − 12 + 4 = 0 ✓\nSince f(2) = 0, **(x − 2) is a factor**.',
          },
          {
            q: 'Express (5x + 1) / [(x+1)(x−2)] in partial fractions.',
            a: '(5x+1)/[(x+1)(x−2)] = A/(x+1) + B/(x−2)\nMultiply through: 5x+1 = A(x−2) + B(x+1)\nx = 2: 11 = 3B → B = 11/3\nx = −1: −4 = −3A → A = 4/3\n**= (4/3)/(x+1) + (11/3)/(x−2)**',
          },
        ],
      },
    ],
  },
  {
    stageId: 'coordinate-geometry',
    stageTitle: 'Stage 2 – Coordinate Geometry & Functions',
    stageDesc: 'Visualising algebra on the coordinate plane and understanding function behaviour.',
    color: '#42A5F5', // medium blue
    topics: [
      {
        id: 'straight-lines',
        title: 'Straight Lines & Coordinate Geometry',
        difficulty: 1,
        concepts: [
          'Equation of a line: y = mx + c and ax + by + c = 0',
          'Gradient, midpoint and distance formulas',
          'Parallel and perpendicular lines',
          'Perpendicular bisector',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/coordinate-geometry/',
        practiceProblems: [
          {
            q: 'Find the equation of the line through (1, 3) with gradient −2.',
            a: 'y − 3 = −2(x − 1)\ny = −2x + 5\n**y = −2x + 5**',
          },
          {
            q: 'Find the perpendicular bisector of the segment from A(2, 4) to B(6, 0).',
            a: 'Midpoint M = (4, 2). Gradient AB = (0−4)/(6−2) = −1. Perpendicular gradient = 1.\ny − 2 = 1(x − 4)\n**y = x − 2**',
          },
        ],
      },
      {
        id: 'functions',
        title: 'Functions',
        difficulty: 2,
        concepts: [
          'Domain and range',
          'Composite functions f(g(x))',
          'Inverse functions f⁻¹(x)',
          'Modulus function y = |f(x)|',
          'Transformations of graphs',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/functions/',
        practiceProblems: [
          {
            q: 'f(x) = 2x + 3 and g(x) = x². Find fg(x).',
            a: 'fg(x) = f(g(x)) = f(x²) = 2x² + 3\n**fg(x) = 2x² + 3**',
          },
          {
            q: 'Find f⁻¹(x) if f(x) = (x − 1) / 3.',
            a: 'Let y = (x − 1)/3 → 3y = x − 1 → x = 3y + 1\nSwap: **f⁻¹(x) = 3x + 1**',
          },
        ],
      },
      {
        id: 'circular-measure',
        title: 'Circular Measure',
        difficulty: 2,
        concepts: [
          'Radian measure: π rad = 180°',
          'Arc length: s = rθ',
          'Area of a sector: A = ½r²θ',
          'Area of a segment',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/circular-measure/',
        practiceProblems: [
          {
            q: 'A sector of radius 5 cm has angle 1.2 rad. Find the arc length and area.',
            a: 'Arc length = rθ = 5 × 1.2 = **6 cm**\nArea = ½r²θ = ½ × 25 × 1.2 = **15 cm²**',
          },
        ],
      },
    ],
  },
  {
    stageId: 'trigonometry',
    stageTitle: 'Stage 3 – Trigonometry',
    stageDesc: 'From basic graphs and identities to advanced techniques used in integration.',
    color: '#1E88E5', // medium-dark blue
    topics: [
      {
        id: 'trig-basics',
        title: 'Trigonometry Basics',
        difficulty: 2,
        concepts: [
          'Exact values: sin/cos/tan of 30°, 45°, 60°',
          'Graphs of sin x, cos x, tan x',
          'Trigonometric equations: general solutions',
          'Pythagorean identity: sin²x + cos²x = 1',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/trigonometry/',
        practiceProblems: [
          {
            q: 'Solve 2sin x = 1 for 0° ≤ x ≤ 360°.',
            a: 'sin x = 0.5 → x = 30° (reference)\n**x = 30° or 150°**',
          },
          {
            q: 'Show that (1 − sin²x) / cos x ≡ cos x.',
            a: 'LHS = cos²x / cos x = **cos x** = RHS ✓',
          },
        ],
      },
      {
        id: 'trig-advanced',
        title: 'Advanced Trigonometry (A2)',
        difficulty: 4,
        concepts: [
          'Sec, cosec, cot and their graphs',
          'Addition formulae: sin(A±B), cos(A±B)',
          'Double angle formulae: sin2A, cos2A',
          'R·sin(x+α) form (harmonic form)',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/trigonometry/',
        practiceProblems: [
          {
            q: 'Express 3sin x + 4cos x in the form R·sin(x + α).',
            a: 'R = √(3² + 4²) = √25 = 5\ntan α = 4/3 → α = 53.13°\n**5·sin(x + 53.1°)**',
          },
          {
            q: 'Prove: (1 + cos2θ) / sin2θ ≡ cot θ',
            a: 'LHS = (1 + (2cos²θ−1)) / 2sinθcosθ\n= 2cos²θ / 2sinθcosθ\n= cosθ / sinθ = **cot θ** ✓',
          },
        ],
      },
    ],
  },
  {
    stageId: 'series-binomial',
    stageTitle: 'Stage 4 – Series & Sequences',
    stageDesc: 'Arithmetic and geometric progressions, then the powerful binomial expansion.',
    color: '#1565C0', // dark blue
    topics: [
      {
        id: 'ap-gp',
        title: 'Arithmetic & Geometric Progressions',
        difficulty: 2,
        concepts: [
          'AP: nth term = a + (n−1)d; sum Sn = n/2(2a + (n−1)d)',
          'GP: nth term = arⁿ⁻¹; sum Sn = a(1−rⁿ)/(1−r)',
          'Convergent GP: S∞ = a/(1−r), |r| < 1',
          'Sigma notation Σ',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/series/',
        practiceProblems: [
          {
            q: 'An AP has first term 5 and common difference 3. Find the 20th term and sum to 20 terms.',
            a: 'T₂₀ = 5 + 19(3) = **62**\nS₂₀ = 20/2(5 + 62) = 10 × 67 = **670**',
          },
          {
            q: 'A GP: 4, 2, 1, ½, … Find the sum to infinity.',
            a: 'a = 4, r = ½ (|r| < 1)\nS∞ = 4/(1 − 0.5) = **8**',
          },
        ],
      },
      {
        id: 'binomial',
        title: 'Binomial Expansion',
        difficulty: 3,
        concepts: [
          'Binomial theorem: (a + b)ⁿ = Σ C(n,r)aⁿ⁻ʳbʳ',
          'Binomial coefficients using nCr = n! / (r!(n−r)!)',
          'Expansion of (1 + x)ⁿ for any n using the general term',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/series/',
        practiceProblems: [
          {
            q: 'Expand (2 + x)⁴ fully.',
            a: '= C(4,0)2⁴ + C(4,1)2³x + C(4,2)2²x² + C(4,3)2x³ + C(4,4)x⁴\n= **16 + 32x + 24x² + 8x³ + x⁴**',
          },
        ],
      },
    ],
  },
  {
    stageId: 'calculus',
    stageTitle: 'Stage 5 – Differentiation',
    stageDesc: 'The core of A-Level maths. Rates of change, optimisation, and curve analysis.',
    color: '#1565C0',
    topics: [
      {
        id: 'differentiation-basics',
        title: 'Differentiation – Basics',
        difficulty: 2,
        concepts: [
          'd/dx (xⁿ) = nxⁿ⁻¹',
          'Tangents and normals at a point',
          'Stationary points and second derivative test',
          'Increasing / decreasing functions',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/differentiation/',
        practiceProblems: [
          {
            q: 'Differentiate f(x) = 3x⁴ − 2x² + 5.',
            a: "**f'(x) = 12x³ − 4x**",
          },
          {
            q: 'Find the stationary points of y = x³ − 3x + 2 and determine their nature.',
            a: "y' = 3x² − 3 = 0 → x² = 1 → x = ±1\ny'' = 6x\nAt x = 1: y'' = 6 > 0 → **minimum** (1, 0)\nAt x = −1: y'' = −6 < 0 → **maximum** (−1, 4)",
          },
        ],
      },
      {
        id: 'differentiation-advanced',
        title: 'Differentiation – Advanced Rules (A2)',
        difficulty: 4,
        concepts: [
          'Chain rule: d/dx[f(g(x))] = f\'(g(x)) · g\'(x)',
          'Product rule: d/dx[uv] = u\'v + uv\'',
          'Quotient rule: d/dx[u/v] = (u\'v − uv\') / v²',
          'Implicit differentiation',
          'd/dx(eˣ) = eˣ, d/dx(ln x) = 1/x',
          'd/dx(sin x) = cos x, d/dx(cos x) = −sin x',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/differentiation/',
        practiceProblems: [
          {
            q: 'Differentiate y = (3x + 1)⁵ using the chain rule.',
            a: 'dy/dx = 5(3x + 1)⁴ · 3 = **15(3x + 1)⁴**',
          },
          {
            q: 'Differentiate y = x²·eˣ using the product rule.',
            a: 'u = x², v = eˣ → u\' = 2x, v\' = eˣ\ndy/dx = 2x·eˣ + x²·eˣ = **eˣ(2x + x²)**',
          },
          {
            q: 'Find dy/dx given x² + y² = 25.',
            a: '2x + 2y(dy/dx) = 0\n**dy/dx = −x / y**',
          },
        ],
      },
    ],
  },
  {
    stageId: 'integration',
    stageTitle: 'Stage 6 – Integration',
    stageDesc: 'Reverse of differentiation: finding areas, volumes and solving differential equations.',
    color: '#0D47A1', // darkest blue
    topics: [
      {
        id: 'integration-basics',
        title: 'Integration – Basics',
        difficulty: 3,
        concepts: [
          '∫xⁿ dx = xⁿ⁺¹/(n+1) + C',
          'Definite integrals and area under a curve',
          'Area between two curves',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-1-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-1/integration/',
        practiceProblems: [
          {
            q: 'Find ∫(4x³ + 6x) dx.',
            a: '= **x⁴ + 3x² + C**',
          },
          {
            q: 'Evaluate ∫₁³ (x² + 1) dx.',
            a: '= [x³/3 + x]₁³ = (9 + 3) − (1/3 + 1) = 12 − 4/3 = **32/3**',
          },
        ],
      },
      {
        id: 'integration-advanced',
        title: 'Integration – Advanced Techniques (A2)',
        difficulty: 5,
        concepts: [
          'Integration by substitution',
          'Integration by parts: ∫u dv = uv − ∫v du',
          '∫eˣ dx = eˣ + C, ∫(1/x) dx = ln|x| + C',
          '∫sin x dx = −cos x + C, ∫cos x dx = sin x + C',
          'Separation of variables for differential equations',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/integration/',
        practiceProblems: [
          {
            q: 'Find ∫x·eˣ dx using integration by parts.',
            a: 'u = x → du = dx; dv = eˣ dx → v = eˣ\n∫x·eˣ dx = x·eˣ − ∫eˣ dx = **x·eˣ − eˣ + C = eˣ(x − 1) + C**',
          },
          {
            q: 'Solve dy/dx = 2xy, given y(0) = 3.',
            a: 'Separate: (1/y)dy = 2x dx\n∫(1/y)dy = ∫2x dx\nln|y| = x² + C\ny = Ae^(x²)\nAt x=0, y=3: A = 3 → **y = 3e^(x²)**',
          },
        ],
      },
    ],
  },
  {
    stageId: 'advanced',
    stageTitle: 'Stage 7 – Advanced Topics (A2)',
    stageDesc: 'Complex numbers, vectors, logarithms, and numerical methods.',
    color: '#0A2D6B',
    topics: [
      {
        id: 'logarithms',
        title: 'Logarithms & Exponentials',
        difficulty: 3,
        concepts: [
          'log laws: log(ab) = log a + log b, log(a/b) = log a − log b',
          'Change of base formula',
          'Solving equations using logs',
          'Natural log ln and e',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/logarithmic-and-exponential-functions/',
        practiceProblems: [
          {
            q: 'Solve 3^(x+1) = 20, giving your answer to 3 s.f.',
            a: '(x+1)·ln 3 = ln 20\nx + 1 = ln20 / ln3 = 2.727\n**x ≈ 1.73**',
          },
        ],
      },
      {
        id: 'vectors',
        title: 'Vectors',
        difficulty: 4,
        concepts: [
          'Vector addition and scalar multiplication',
          'Position vectors and displacement',
          'Magnitude and unit vectors',
          'Equation of a straight line in 3D: r = a + λb',
          'Angle between two vectors using dot product',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/vectors/',
        practiceProblems: [
          {
            q: 'Find the angle between vectors a = (1, 2, −1) and b = (3, 0, 1).',
            a: 'a·b = 3 + 0 − 1 = 2\n|a| = √6, |b| = √10\ncos θ = 2/(√6·√10) = 2/√60\n**θ = cos⁻¹(2/√60) ≈ 75.0°**',
          },
        ],
      },
      {
        id: 'complex-numbers',
        title: 'Complex Numbers',
        difficulty: 5,
        concepts: [
          'i = √(−1), z = a + bi',
          'Modulus |z| = √(a² + b²) and argument arg(z)',
          'Argand diagram',
          'Complex conjugate z* = a − bi',
          'Solving quadratics with complex roots',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/complex-numbers/',
        practiceProblems: [
          {
            q: 'Solve x² + 2x + 5 = 0.',
            a: 'x = (−2 ± √(4−20)) / 2 = (−2 ± √(−16)) / 2\n= (−2 ± 4i) / 2\n**x = −1 + 2i or x = −1 − 2i**',
          },
        ],
      },
      {
        id: 'numerical-methods',
        title: 'Numerical Methods',
        difficulty: 4,
        concepts: [
          'Locating roots by sign change',
          'Iterative methods: xₙ₊₁ = g(xₙ)',
          'Convergence / divergence conditions',
        ],
        tutorialLink: 'https://www.examsolutions.net/a-level-maths/cie/pure-maths-3-tutorials/',
        practiceLink: 'https://www.physicsandmathstutor.com/maths-revision/a-level-cie/pure-3/numerical-solutions-of-equations/',
        practiceProblems: [
          {
            q: 'Show there is a root of x³ − 2x − 1 = 0 between x = 1 and x = 2.',
            a: 'f(1) = 1 − 2 − 1 = −2 < 0\nf(2) = 8 − 4 − 1 = 3 > 0\nSign change → **root exists in (1, 2)** by IVT.',
          },
        ],
      },
    ],
  },
];

export const getTopicById = (topicId) => {
  for (const stage of CAMBRIDGE_TOPIC_PATH) {
    const found = stage.topics.find((t) => t.id === topicId);
    if (found) return { ...found, stageColor: stage.color, stageTitle: stage.stageTitle };
  }
  return null;
};

// ── Keep old year-based data for NCEA / IB ─────────────────────────
export const CURRICULUM_DATA = {
  cambridge: {
    12: {
      title: 'AS Level Mathematics',
      subtitle: 'Cambridge International',
      description: 'The primary half of the A-Level course. Covers Pure Mathematics 1 and Probability & Statistics 1.',
      goals: [
        'Master advanced algebra, functions, and coordinate geometry.',
        'Understand the foundational theories of Calculus (Differentiation & Integration).',
        'Apply mathematical models to perform data analysis (Statistics).',
      ],
      topics: [],
      pastPapers: [],
    },
    13: {
      title: 'A2 Level Mathematics',
      subtitle: 'Cambridge International',
      description: 'The second half of the A-Level course. Covers Pure Mathematics 3 and Probability & Statistics 2.',
      goals: [
        'Master complex algebra, logarithms, and advanced trigonometry.',
        'Deepen understanding of Calculus with integration by parts and differential equations.',
        'Perform complex vector operations and hypothesis testing.',
      ],
      topics: [],
      pastPapers: [],
    },
  },
};

export const getCurriculumData = (system, year) =>
  CURRICULUM_DATA[system]?.[year] ?? null;
