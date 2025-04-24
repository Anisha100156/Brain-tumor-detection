// TestimonialSlider.tsx
"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";

const testimonials = [
  {
    image: "/test1.jpeg",
    quote:
      "I had been putting off seeing a doctor, brushing off symptoms as stress. But this site was so easy to use — I just uploaded my scan and within minutes, I had a result. Seeing a possible issue made it real for me, and I finally made the appointment I’d been avoiding. That small push made a huge difference.",
    name: "Shelley",
  },
  {
    image: "/test2.jpeg",
    quote: "For weeks, I had been living in fear. I stumbled on this platform while looking for online support, and decided to try it. When I saw 'No tumor detected,' I felt immediate relief. Of course, I still followed up with a doctor, but having that reassurance helped me stay calm. It’s like having a friend that understands your fear.",
    name: "Angela",
  },
  {
    image: "/test3.jpeg",
    quote: "My father was too scared to go to a hospital. We convinced him to upload his scan here first, just to get an idea. When the model detected a possible glioma, we used that information to persuade him to get checked. Doctors confirmed it — but thankfully, it was early. Your website helped us overcome fear with facts.",
    name: "Mark",
  },
  {
    image: "/test4.jpeg",
    quote: "Being unsure about my health is terrifying. I was experiencing symptoms but didn't want to panic or self-diagnose. Uploading my brain scan here gave me a clear, easy-to-understand result. It wasn’t a replacement for medical advice, but it gave me confidence to take action and talk to a professional. That peace of mind? Priceless.",
    name: "Lucas",
  },
];

export default function TestimonialSlider() {
  const [currentIndex, setCurrentIndex] = useState(0);
 const intervalRef = useRef(null);

  const prevSlide = () => {
    setCurrentIndex((prev) => (prev === 0 ? testimonials.length - 1 : prev - 1));
  };

  const nextSlide = () => {
    setCurrentIndex((prev) => (prev === testimonials.length - 1 ? 0 : prev + 1));
  };

  useEffect(() => {
    intervalRef.current = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % testimonials.length);
    }, 2000);
    return () => intervalRef.current && clearInterval(intervalRef.current);
  }, []);

  return (
    <div className="relative bg-gradient-to-b from-[#c8d8ef] to-[#bcd7d1] py-24 overflow-hidden">
      <h2 className="text-4xl text-center font-bold text-[#3B4461] mb-16">Our Stories</h2>

      {/* Main Slider */}
      <div className="relative w-[720px] mx-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentIndex}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.5 }}
            className="flex w-full bg-white rounded-xl shadow-2xl overflow-hidden"
          >
            <div className="w-1/2 h-[400px] relative">
              <Image
                src={testimonials[currentIndex].image}
                alt="Testimonial"
                layout="fill"
                objectFit="cover"
              />
            </div>
            <div className="w-1/2 bg-[#3ca48d] text-white p-8 flex flex-col justify-center">
              <p className="text-lg leading-relaxed font-semibold">
                {testimonials[currentIndex].quote}
              </p>
              <p className="mt-4 text-sm">— shared by {testimonials[currentIndex].name}</p>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Arrows */}
        <button
          onClick={prevSlide}
          className="absolute left-[-48px] top-1/2 -translate-y-1/2 bg-white rounded-full p-2 shadow-lg hover:bg-gray-100"
        >
          <ChevronLeft size={24} className="text-[#3ca48d]" />
        </button>
        <button
          onClick={nextSlide}
          className="absolute right-[-48px] top-1/2 -translate-y-1/2 bg-white rounded-full p-2 shadow-lg hover:bg-gray-100"
        >
          <ChevronRight size={24} className="text-[#3ca48d]" />
        </button>
      </div>

      {/* Thumbnail Slider */}
      <div className="mt-16 flex justify-center gap-6 px-8 overflow-x-auto">
        {testimonials.map((item, index) => (
          <div
            key={index}
            className={`w-[180px] h-[120px] rounded-lg overflow-hidden transition-all duration-300 ease-in-out cursor-pointer ${
              index === currentIndex ? "opacity-100 scale-105" : "opacity-40"
            }`}
            onClick={() => setCurrentIndex(index)}
          >
            <Image
              src={item.image}
              alt="Thumbnail"
              width={180}
              height={120}
              className="object-cover w-full h-full"
            />
          </div>
        ))}
      </div>

      {/* CTA Button */}
      <div className="mt-12 text-center">
        <button className="bg-white text-[#3ca48d] font-medium px-6 py-2 rounded-full shadow hover:bg-gray-100 transition">
          View all Patient Stories
        </button>
      </div>
    </div>
  );
}
