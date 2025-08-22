import React from 'react'

interface LabelProps {
  children: React.ReactNode
  className?: string
  htmlFor?: string
}

export const Label: React.FC<LabelProps> = ({ children, className = "", htmlFor }) => {
  return (
    <label htmlFor={htmlFor} className={`text-sm font-medium leading-none ${className}`}>
      {children}
    </label>
  )
}

export default Label