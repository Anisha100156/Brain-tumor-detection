import React from "react";
import { useRouter } from "next/navigation";
import { Play } from "lucide-react";
import { TextGenerateEffect } from "./text-generate";
import { TextGenerate } from "./text-gen";
const help="How can We help You"
const HelpSection = () => {
  const router = useRouter();
  return (
    <section className="relative py-16 px-6 bg-white">
  
      <div className="absolute left-0 bottom-0 opacity-5 z-0">
       
        <img src="	https://149587026.v2.pressablecdn.com/wp-content/t…heme/assets/images/brain-graphic-background-c.svg" alt="Brain Icon" className="w-64 h-64" />
      </div>

      <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center relative z-10 gap-10">
        
        <div className="flex-1 text-left">
         <TextGenerate words={help}/>
          <p className="text-[#212636] mb-4 leading-relaxed font-serif text-lg">
            Brain Tumor Network is a trusted source for reliable information about
            primary brain tumors for patients of all ages. Our dedicated,
            neuro-oncology navigators help you find answers to the difficult questions
            you may be asking yourself:
          </p>
          <ul className="list-disc list-inside text-[#212636] mb-6 font-serif text-lg leading-relaxed">
  <li>Where do I begin?</li>
  <li>Should I get a second opinion?</li>
  <li>What are my treatment options?</li>
  <li>What can I expect next?</li>
</ul>

          <button className="shadow-[inset_0_0_0_2px_#616467] text-white px-12 py-4 rounded-full tracking-widest uppercase font-bold bg-[#164F90] hover:bg-[#306E8B] hover:text-white dark:text-[#5AB99F] transition duration-200" onClick={() => router.push("/tumor-detection")}>
  Take the first step
</button>
        </div>

        {/* Right Content - Image with Play Button */}
        <div className="flex-1 relative">
          <img
            src="/images.jpg"
            alt="Couple on bench"
            className="rounded-lg w-full h-auto object-cover"
          />
          <button className="absolute inset-0 flex justify-center items-center">
            
          </button>
        </div>
      </div>
    </section>
  );
};

export default HelpSection;
