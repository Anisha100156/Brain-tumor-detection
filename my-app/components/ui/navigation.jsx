import React from "react";
import { Brain, BookOpen, Headphones, FileText } from "lucide-react";

const Navigation = () => {
  const services = [
    {
      icon: <Brain className="w-10 h-10 text-white" />,
      title: "Understand Your Diagnosis",
    },
    {
      icon: <BookOpen className="w-10 h-10 text-white" />,
      title: "Explore Treatment Options",
    },
    {
      icon: <Headphones className="w-10 h-10 text-white" />,
      title: "Receive Emotional Support",
    },
    {
      icon: <FileText className="w-10 h-10 text-white" />,
      title: "Get Medical Records Assistance",
    },
  ];

  return (
    <div className="relative w-full">
      {/* Curved Top Edge */}
      <div className="w-full overflow-hidden leading-[0] rotate-180">
        <svg viewBox="0 0 1440 100" className="w-full h-24" preserveAspectRatio="none">
          <path
            d="M0,32 C360,100 1080,0 1440,48 L1440,0 L0,0 Z"
            fill="#74B1AF"
          ></path>
        </svg>
      </div>

      {/* Main Content */}
      <div className="bg-gradient-to-br from-[#74B1AF] via-[#5AB99F] to-[#403D64] py-10 text-white text-center px-6">
        <h2 className="text-3xl font-bold mb-6">Our Navigation Services</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6 justify-center items-start max-w-5xl mx-auto">
          {services.map((service, index) => (
            <div key={index} className="flex flex-col items-center">
              <div className="bg-[#403D64] rounded-full p-4 mb-3">{service.icon}</div>
              <p className="font-semibold text-base leading-snug max-w-[10rem]">
                {service.title}
              </p>
            </div>
          ))}
        </div>

        <button className="mt-6 px-6 py-3 bg-white text-[#306E8B] font-semibold rounded-full shadow-lg hover:bg-gray-100 transition">
          Explore Navigation Services
        </button>

        <div className="mt-6 max-w-3xl mx-auto text-sm sm:text-base text-white font-light">
          <h3 className="font-semibold text-base mt-4 mb-1">
            What is a patient navigator?
          </h3>
          <p>
            Our brain tumor patient navigators are professionals guiding patients and
            caregivers through every phase of brain tumor care with personalized
            one-on-one options, information, and support.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Navigation;
