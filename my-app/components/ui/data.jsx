"use client";

import CountUp from "react-countup";
import { useInView } from "react-intersection-observer";

const stats = [
  {
    icon: "🤝",
    value: 3089,
    suffix: "+",
    description: "Patients who have received free, personalized navigation services",
  },
  {
    icon: "🧠",
    value: 92,
    suffix: "%",
    description: "Patients who would recommend the Brain Tumor Network to others",
  },
  {
    icon: "⭐",
    value: 483,
    suffix: "+",
    description: "Relationships built with national brain tumor centers of excellence and regional treatment centers across the country",
  },
  {
    icon: "👫",
    value: 28,
    suffix: "+",
    description: "Years of treatment-related and survivorship experiences of founders Susan and Rick Sontag",
  },
  {
    icon: "👩‍⚕️",
    value: 10,
    suffix: "+",
    description: "Years providing direct, personalized support to adult primary brain tumor patients and caregivers",
  },
  {
    icon: "🧑‍🔬",
    value: 10,
    suffix: "+",
    description: "Years building relationships within the neuro-oncology community",
  },
];

export default function OurExperience() {
  const { ref, inView } = useInView({
    triggerOnce: true,
    threshold: 0.3,
  });

  return (
    <section ref={ref} className="bg-gradient-to-r from-[#3B4461] to-[#4D577B] text-white py-20 px-8 md:px-24">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4">Our Experience</h2>
        <p className="max-w-3xl mx-auto text-lg">
          At its heart, our organization is committed to providing a community of support for individuals and families,
          focusing on survivorship. We are most proud of the relationships we have built along the way and the
          individual lives we continue to impact as we fulfill our mission.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-12 text-center">
        {stats.map((stat, index) => (
          <div key={index} className="flex flex-col items-center">
            <div className="text-5xl mb-3">{stat.icon}</div>
            <h3 className="text-3xl font-bold">
              {inView ? <CountUp end={stat.value} duration={2} /> : 0}
              {stat.suffix}
            </h3>
            <p className="mt-2 max-w-xs">{stat.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
