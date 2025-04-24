"use client";

import { useEffect, useRef } from "react";
import { useInView } from "react-intersection-observer";
import { animate, useMotionValue } from "framer-motion";
import { Chart } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(ArcElement, Tooltip, Legend);

const stats = [
  {
    label:
      "Patients who have received free, personalized navigation services",
    value: 3089,
    icon: (
      <img
        src="https://www.braintumornetwork.org/wp-content/uploads/2021/02/our-impact_icon6.svg"
        alt="Impact Icon"
        className="w-16 h-16 mb-2"
      />
    ),
  },
  {
    label: "Patients who would recommend the Brain Tumor Network to others",
    value: 92,
    suffix: "%",
    icon: (
      <img
        src="https://www.braintumornetwork.org/wp-content/uploads/2021/02/our-impact_icon2.svg"
        alt="Impact Icon"
        className="w-16 h-16 mb-2"
      />
    ),
  },
  {
    label:
      "Relationships with national brain tumor centers across the country",
    value: 483,
    icon: (
      <img
        src="https://www.braintumornetwork.org/wp-content/uploads/2021/02/our-impact_icon5.svg"
        alt="Impact Icon"
        className="w-16 h-16 mb-2"
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
  const data = {
    labels: [
      "Free Navigation Services",
      "Would Recommend BTN",
      "National Brain Tumor Centers",
    ],
    datasets: [
      {
        label: "2023",
        data: [3000, 80, 400],
        backgroundColor: ["#6DA5C0", "#8AB6D6", "#42566B"],
      },
      {
        label: "2024",
        data: [3089, 92, 483],
        backgroundColor: ["#4D90A4", "#699CC0", "#2E3D4E"],
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom",
        labels: {
          color: "white",
        },
      },
    },
  };

  return (
    <div className="w-full max-w-lg h-96 mx-auto">
      <Chart type="bar" data={data} options={options} />
    </div>
  );
};

const ExperienceSection = () => {
  return (
    <div className="relative w-full">
      <div className="w-full overflow-hidden leading-[0] rotate-180">
        <svg
          viewBox="0 0 1440 100"
          className="w-full h-20"
          preserveAspectRatio="none"
        >
          <path
            d="M0,32 C360,100 1080,0 1440,48 L1440,0 L0,0 Z"
            fill="#42566B"
          ></path>
        </svg>
      </div>

      <section
        className="py-12 text-white"
        style={{
          background:
            "linear-gradient(to bottom, #42566B, #44586E, #515274, #54567A)",
        }}
      >
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-8">Our Experience</h2>

          <div className="flex flex-col lg:flex-row items-center justify-between gap-12">
            <div className="grid gap-8 grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 flex-1">
              {stats.map((stat, index) => (
                <div key={index} className="flex items-center space-x-4">
                  <div className="shrink-0">{stat.icon}</div>
                  <div>
                    <div className="text-2xl font-bold">
                      <Counter target={stat.value} suffix={stat.suffix || ""} />+
                    </div>
                    <p className="text-sm">{stat.label}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex-1 flex justify-center items-center">
              <ChartDisplay />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ExperienceSection;
