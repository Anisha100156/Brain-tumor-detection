import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronUp } from 'lucide-react';

const symptomsData = [
  {
    title: 'Seizures',
    content: 'Seizures may occur when the brain\'s normal electrical activity is disrupted by a tumor. These can vary in severity and presentation, ranging from full-body convulsions to subtle muscle twitches or temporary confusion.Can occur when abnormal electrical activity is triggered in the brain due to the tumor’s interference. These seizures can manifest as full-body convulsions, brief episodes of staring, sudden jerking of limbs, or even subtle twitching. Some individuals may also experience strange sensations or visual changes before a seizure, known as auras.'
  },
  {
    title: 'Headaches',
    content: `Headaches are a common ailment in healthy people and can be the result of many everyday causes.
Headaches that are associated with brain tumors, however, typically have clear differences from "regular" headaches. Headaches caused by brain tumors usually:

- Are worst early in the morning
- Increase in frequency and intensity over time
- Are associated with repeated or prolonged vomiting
- Can be made worse by coughing, sneezing, or straining
- Don't respond to typical over-the-counter medications`
  },
  {
    title: 'Speech and Language Difficulties',
    content: 'Depending on the tumor’s location, individuals may have difficulty speaking, understanding language, or forming coherent sentences. May arise when tumors are located in areas responsible for communication, typically in the left hemisphere of the brain. This can result in challenges such as struggling to find the right words, slurred or slowed speech, and difficulty understanding or forming sentences. Reading and writing abilities may also be impaired.'
  },
  {
    title: 'Memory Loss',
    content: 'Short-term memory loss or confusion may occur, especially if the tumor affects the frontal or temporal lobes of the brain. This can include short-term memory problems, difficulty concentrating, confusion, and trouble recalling important details or events. Individuals might find themselves feeling disoriented or struggling with tasks that were once familiar.'
  },
  {
    title: 'Mass Effect',
    content: 'As the tumor grows, it can create pressure on surrounding brain tissue, resulting in changes in mental function, behavior, and alertness.the pressure it exerts on surrounding brain tissue—can lead to broader neurological and behavioral changes. These include personality shifts, drowsiness, reduced alertness, vision disturbances, and problems with balance or coordination. As the tumor expands, it can significantly impact overall brain function and quality of life.'


  }
];

export default function BrainTumorSymptoms() {
  const [activeIndex, setActiveIndex] = useState(null);

  const toggleIndex = (index) => {
    setActiveIndex(index === activeIndex ? null : index);
  };

  return (
    <div className="relative w-full  ">
    
  <svg viewBox="0 0 1440 100" className="w-full h-24" preserveAspectRatio="none">
  <defs>
    <linearGradient id="symptomsWaveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stopColor="#9DBFCA" />
      <stop offset="35%" stopColor="#A2ADCB" />
      <stop offset="60%" stopColor="#AFACD2" />
      <stop offset="75%" stopColor="#92A0C8" />
      <stop offset="100%" stopColor="#8EA2C7" />
    </linearGradient>
  </defs>
  <g transform="scale(1, -1) translate(0, -100)">
    <path
      d="M0,32 C360,100 1080,0 1440,48 L1440,0 L0,0 Z"
      fill="url(#symptomsWaveGradient)"
    />
  </g>
</svg>

    <div className="max-w-9xl  relative top-[-20] mx-auto pl-60 pr-60 pt-20  bg-gradient-to-br from-[#9DBFCA] via-[#A2ADCB] via-[#AFACD2] via-[#92A0C8] to-[#8EA2C7] min-h-[90vh]">
    <h1 className="text-5xl font-bold mb-8 font-serif text-[#000080]">Common Brain Tumor Signs and Symptoms</h1>

      {symptomsData.map((item, index) => (
        <div key={index} className="mb-4 border-2 border-gray-700 rounded-md overflow-hidden">
          <button
            className="w-full flex justify-between items-center text-left px-4 py-3 font-semibold text-lg bg-white hover:bg-gray-100 transition-colors duration-300"
            onClick={() => toggleIndex(index)}
          >
            <span>{item.title}</span>
            {activeIndex === index ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
          </button>
          <AnimatePresence initial={false}>
            {activeIndex === index && (
              <motion.div
                key={index}
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.4, ease: 'easeInOut' }}
                className="overflow-hidden bg-blue-900 text-white px-6 py-4 whitespace-pre-line"
              >
                {item.content}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ))}
      
    </div>
    </div>
  );
}
