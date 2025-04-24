const Wavesx = () => {
  return (
    <div className="relative w-full h-[320px] text-white">
      {/* Mission Quote */}
      <div className="absolute z-10 top-38 left-70  max-w-4xl">
        <h2 className="text-green-400 font-bold text-lg md:text-2xl mb-2">
        The symptoms of a brain tumor are rarely the same for everyone. They depend not only on the size of the tumor, but more importantly on where it grows and how much pressure it puts on the surrounding brain tissues
        </h2>
       

      </div>

      {/* SVG Waves */}
      <svg
        viewBox="0 0 1440 320"
        className="absolute bottom-0 w-full transform scale-x-[-1]"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          fill="#64587A"
          fillOpacity="1"
          d="M0,160L80,170.7C160,181,320,203,480,186.7C640,171,800,117,960,101.3C1120,85,1280,107,1360,117.3L1440,128V320H0Z"
        ></path>
        <path
          fill="#698A91"
          fillOpacity="1"
          d="M0,192L80,186.7C160,181,320,171,480,149.3C640,128,800,96,960,122.7C1120,149,1280,235,1360,245.3L1440,256V320H0Z"
        ></path>
        <defs>
          <linearGradient id="blueGradient" gradientTransform="rotate(0)">
            <stop offset="0%" stopColor="#516BAA" />
            <stop offset="50%" stopColor="#184D8C" />
            <stop offset="100%" stopColor="#307695" />
          </linearGradient>
        </defs>
        <path
          fill="url(#blueGradient)"
          fillOpacity="1"
          d="M0,224L80,213.3C160,203,320,181,480,154.7C640,128,800,96,960,106.7C1120,117,1280,171,1360,192L1440,224V320H0Z"
        ></path>
      </svg>
    </div>
  );
};

export default Wavesx;
