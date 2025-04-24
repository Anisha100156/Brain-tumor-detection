"use client";

import { useEffect, useRef } from "react";
import { useInView } from "react-intersection-observer";
import { animate, useMotionValue } from "framer-motion";
import { Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

const stats = [
  {
    label: "Users are returning - A strong sign of satisfaction and platform reliability",
    value: 54,
    suffix: "%",
    icon: (
      <img
        src="https://www.braintumornetwork.org/wp-content/uploads/2021/02/our-impact_icon6.svg"
        alt="Impact Icon"
        className="w-16 h-16 mb-1"
      />
    ),
  },
  {
    label: "Our platform has welcomed users from 21 different countries, reflecting strong global curiosity around AI-driven brain tumor detection",
    value: 15,
    suffix: "%",
    icon: (
      <img
        src="https://www.braintumornetwork.org/wp-content/uploads/2021/02/our-impact_icon2.svg"
        alt="Impact Icon"
        className="w-16 h-16 mb-1"
      />
    ),
  },
  {
    label: "Early tumor detection cases where our model flagged potential abnormalities - enabling proactive care",
    value: 83,
    suffix: "%",
    icon: (
      <img
        src="https://www.braintumornetwork.org/wp-content/uploads/2021/02/our-impact_icon5.svg"
        alt="Impact Icon"
        className="w-16 h-16 mb-1"
      />
    ),
  },
  {
    label: "A steady stream of new users continues to trust our website for initial brain health checks",
    value: 70,
    suffix: "%",
    icon: (
      <img
        src="https://www.braintumornetwork.org/wp-content/uploads/2021/02/our-impact_icon1.svg"
        alt="Impact Icon"
        className="w-16 h-16 mb-1"
      />
    ),
  },
  {
    label: "Thousands have accessed our platform, reflecting growing interest in AI-assisted brain tumor screening",
    value: 1000,
    suffix: "+",
    icon: (
      <img
        src="https://www.braintumornetwork.org/wp-content/uploads/2021/02/our-impact_icon3.svg"
        alt="Impact Icon"
        className="w-16 h-16 mb-1"
      />
    ),
  },
  {
    label: "Hundreds of predictions have been made helping users make informed next steps in their health journey",
    value: 100,
    suffix: "+",
    icon: (
      <img
        src="https://www.braintumornetwork.org/wp-content/uploads/2021/02/our-impact_icon4.svg"
        alt="Impact Icon"
        className="w-16 h-16 mb-1"
      />
    ),
  },
];

const Counter = ({ target, suffix = "" }) => {
  const count = useMotionValue(0);
  const { ref, inView } = useInView({ triggerOnce: true });
  const spanRef = useRef(null);

  useEffect(() => {
    if (inView) {
      const controls = animate(count, target, {
        duration: 2,
        onUpdate(value) {
          if (spanRef.current) {
            spanRef.current.textContent = `${Math.floor(value)}${suffix}`;
          }
        },
      });
      return () => controls.stop();
    }
  }, [inView, target, suffix]);

  return (
    <span ref={ref}>
      <span ref={spanRef}>0{suffix}</span>
    </span>
  );
};

const ChartDisplay = () => {
  const datasetLabels = [
    {
      layer: "Users",
      labels: ["Visited", "Revisited", "New Users"],
      colors: ["#AD95CD", "#64587A", "#E0B0FF"],
    },
    {
      layer: "Type of tumor detected",
      labels: ["Glioma", "Meningioma", "Pituitary"],
      colors: ["#6495ED", "#ADD8E6", "#87CEFA"],
    },
    {
      layer: "User satisfaction Level",
      labels: ["Satisfied", "Dissatisfied", "Happy"],
      colors: ["#3CB371", "#90D6C2", "#62BCA3"],
    },
  ];

  const data = {
    labels: [],
    datasets: datasetLabels.map((set) => ({
      label: set.layer,
      data: [65, 10, 25],
      backgroundColor: set.colors,
      borderColor: "#ffffff",
      borderWidth: 2,
      customLabels: set.labels,
    })),
  };

  const options = {
    responsive: true,
    cutout: "20%",
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function (tooltipItem) {
            const dataset = data.datasets[tooltipItem.datasetIndex];
            const label =
              dataset.customLabels?.[tooltipItem.dataIndex] || "Data";
            const value = dataset.data[tooltipItem.dataIndex];
            return `${dataset.label} - ${label}: ${value}`;
          },
        },
      },
    },
  };

  return (
    <div className="w-full max-w-xs mx-auto lg:mx-0 flex flex-col items-center">
      <Pie data={data} options={options} />
      <div className="mt-4 space-y-2 text-sm text-white">
        {datasetLabels.map((set, index) => (
          <div key={index}>
            <p className="font-semibold mb-1">{set.layer}</p>
            <div className="flex flex-wrap gap-2">
              {set.labels.map((label, idx) => (
                <div key={idx} className="flex items-center gap-1">
                  <span
                    className="w-3 h-3 inline-block rounded-full"
                    style={{ backgroundColor: set.colors[idx] }}
                  ></span>
                  <span>{label}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const ExperienceSection = () => {
  const { ref: chartRef, inView: chartInView } = useInView({
    triggerOnce: true,
    threshold: 0.2,
  });

  return (
    <div className="relative w-full -top-32">
      <div className="w-full overflow-hidden leading-[0] rotate-180">
        <svg
          viewBox="0 0 1440 100"
          className="w-full h-16"
          preserveAspectRatio="none"
        >
          <path
            d="M0,32 C360,100 1080,0 1440,48 L1440,0 L0,0 Z"
            fill="#42566B"
          ></path>
        </svg>
      </div>

      <section
        className="py-8 text-white"
        style={{
          background:
            "linear-gradient(to bottom, #42566B, #44586E, #515274, #54567A)",
        }}
      >
        <div className="max-w-7xl px-4 mx-auto">
          <h2 className="text-3xl font-semibold mb-6 text-center">
            Our Experience
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            {/* Stats */}
            <div className="col-span-12 lg:col-span-7 grid grid-cols-1 md:grid-cols-2 gap-6">
              {stats.map((stat, index) => (
                <div key={index} className="flex flex-col items-center text-center">
                  <div className="text-4xl mb-1">{stat.icon}</div>
                  <div className="text-2xl font-bold">
                    <Counter target={stat.value} suffix={stat.suffix || ""} />
                  </div>
                  <p className="text-sm mt-1 px-2 leading-snug">{stat.label}</p>
                </div>
              ))}
            </div>

            {/* Chart (Lazy-loaded on scroll) */}
            <div
              ref={chartRef}
              className="col-span-12 lg:col-span-5 flex justify-center lg:justify-end"
            >
              {chartInView && <ChartDisplay />}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ExperienceSection;
