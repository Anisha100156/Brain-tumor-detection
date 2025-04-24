'use client'

import { motion } from 'framer-motion'

export default function Loading() {
  return (
    <motion.div
      className="flex items-center justify-center min-h-screen bg-white"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="w-16 h-16 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
    </motion.div>
  )
}
