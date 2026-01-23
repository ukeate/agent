import React from 'react'

interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'destructive' | 'warning' | 'success'
  children: React.ReactNode
}

interface AlertDescriptionProps
  extends React.HTMLAttributes<HTMLParagraphElement> {
  children: React.ReactNode
}

export const Alert: React.FC<AlertProps> = ({
  variant = 'default',
  children,
  className = '',
  ...props
}) => {
  const variants = {
    default: 'bg-blue-50 border-blue-200 text-blue-800',
    destructive: 'bg-red-50 border-red-200 text-red-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    success: 'bg-green-50 border-green-200 text-green-800',
  }

  return (
    <div
      className={`border rounded-md p-4 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </div>
  )
}

export const AlertDescription: React.FC<AlertDescriptionProps> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <p className={`text-sm ${className}`} {...props}>
      {children}
    </p>
  )
}
