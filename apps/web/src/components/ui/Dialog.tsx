import React from 'react'

interface DialogProps {
  open?: boolean
  isOpen?: boolean
  onOpenChange?: (open: boolean) => void
  onClose?: () => void
  maxWidth?: string
  children: React.ReactNode
}

export const Dialog: React.FC<DialogProps> = ({
  open,
  isOpen,
  onOpenChange,
  onClose,
  maxWidth = 'md',
  children,
}) => {
  const isVisible = open || isOpen
  if (!isVisible) return null

  const handleClose = () => {
    if (onClose) onClose()
    if (onOpenChange) onOpenChange(false)
  }

  const maxWidthClass =
    {
      sm: 'max-w-sm',
      md: 'max-w-md',
      lg: 'max-w-lg',
      xl: 'max-w-xl',
      '2xl': 'max-w-2xl',
      '3xl': 'max-w-3xl',
      '4xl': 'max-w-4xl',
      '5xl': 'max-w-5xl',
      '6xl': 'max-w-6xl',
    }[maxWidth] || 'max-w-md'

  return (
    <div className="fixed inset-0 z-50">
      <div
        className="fixed inset-0 bg-black bg-opacity-50"
        onClick={handleClose}
      />
      <div className="fixed inset-0 flex items-center justify-center p-4 pointer-events-none">
        <div
          className={`pointer-events-auto bg-white rounded-lg shadow-lg ${maxWidthClass} w-full max-h-[90vh] overflow-y-auto`}
        >
          {children}
        </div>
      </div>
    </div>
  )
}

interface DialogContentProps {
  children: React.ReactNode
  maxWidth?: string
}

export const DialogContent: React.FC<DialogContentProps> = ({
  children,
  maxWidth = 'md',
}) => {
  const maxWidthClass =
    {
      sm: 'max-w-sm',
      md: 'max-w-md',
      lg: 'max-w-lg',
      xl: 'max-w-xl',
      '2xl': 'max-w-2xl',
      '3xl': 'max-w-3xl',
      '4xl': 'max-w-4xl',
      '5xl': 'max-w-5xl',
      '6xl': 'max-w-6xl',
    }[maxWidth] || 'max-w-md'

  return (
    <div
      className={`bg-white rounded-lg shadow-lg ${maxWidthClass} w-full mx-4 max-h-[90vh] overflow-y-auto`}
    >
      {children}
    </div>
  )
}

export const DialogHeader: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return <div className="mb-4">{children}</div>
}

export const DialogTitle: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return <h2 className="text-lg font-semibold">{children}</h2>
}

export const DialogDescription: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return <p className="text-sm text-gray-600">{children}</p>
}

export const DialogFooter: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return <div className="mt-6 flex justify-end space-x-2">{children}</div>
}

export const DialogTrigger: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return <>{children}</>
}
