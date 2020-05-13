% Corporate beamer-template for KU Leuven en Kulak
% Stijn Rebry - september 2012
% Tips en hints voor een beter resultaat altijd welkom
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{kulakbeamer}[2015/09/22 Beamer class for KU Leuven]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax
\LoadClass{beamer}
\RequirePackage{times}

% Definitie kleuren
\definecolor{kulblauw}{RGB}{0,64,122}
\colorlet{kulblauw10} {kulblauw!10!white}
\definecolor{kulappelblauwzeegroen}{RGB}{20,116,146}
\colorlet{kulappelblauwzeegroen10} {kulappelblauwzeegroen!10!white}
\definecolor{kuloranje}{RGB}{245,131,87}
\colorlet{kuloranje10} {kuloranje!10!white}
\definecolor{kulgroen}{RGB}{82,189,236}
\colorlet{kulgroen10} {kulgroen!10!white}

% Opties instellen
\DeclareOptionBeamer{kul}{
  \newcommand{\Background}{\includegraphics[width=\paperwidth,height=\paperheight]{kuleuvenBackground}}
  \newcommand{\Titlepage}{\includegraphics[width=\paperwidth,height=\paperheight]{kuleuvenTitlepage}}
  \newcommand{\Outline}{\includegraphics[width=\paperwidth,height=\paperheight]{kuleuvenOutline}}
  \newcommand{\Logo}{\includegraphics[height=.8cm]{kuleuvenLogo}}
  }
\DeclareOptionBeamer{kulak}{
  \newcommand{\Background}{\includegraphics[width=\paperwidth,height=\paperheight]{kulakBackground}}
  \newcommand{\Titlepage}{\includegraphics[width=\paperwidth,height=\paperheight]{kulakTitlepage}}
  \newcommand{\Outline}{\includegraphics[width=\paperwidth,height=\paperheight]{kulakOutline}}
  \newcommand{\Logo}{\includegraphics[height=.8cm]{kulakLogo}}
  }
\ProcessOptionsBeamer


\mode<beamer>
\useinnertheme{default}
\usebackgroundtemplate{\Background}
\newenvironment{titleframe}
  {\bgroup\usebackgroundtemplate{\Titlepage}\begin{frame}[plain]}
  {\end{frame}\egroup}
\newenvironment{outlineframe}[1][]
  {\bgroup\usebackgroundtemplate{\Outline}\begin{frame}[plain]{\textcolor{white}{#1}}}
  {\end{frame}\egroup}

%\begin{outlineframe}[\parbox{.38\textwidth}{~}\parbox{.62\textwidth}{Outline}]
%\parbox{.38\textwidth}{~}\parbox{.62\textwidth}{\setlength{\parskip}{2ex}\tableofcontents}
%\end{outlineframe}


\setbeamercolor{frametitle} {fg=kulblauw}%
\setbeamercolor{block title} {use=structure,fg=white,bg=kulappelblauwzeegroen}
\setbeamercolor{block title alerted} {use=alerted text,fg=white,bg=kuloranje}
\setbeamercolor{block title example} {use=example text,fg=white,bg=kulgroen}
\setbeamercolor{block body} {fg=kulblauw,use=block title,bg=kulappelblauwzeegroen10}
\setbeamercolor{block body alerted} {fg=kulblauw,use=block title alerted,bg=kuloranje10}
\setbeamercolor{block body example} {fg=kulblauw,use=block title example,bg=kulgroen10}
\setbeamercolor{sidebar canvas right}{fg=kulblauw,use=block title example,bg=kulblauw10}
\setbeamercolor{title}{fg=white}
\setbeamercolor{title page}{fg=white}

\defbeamertemplate*{title page}{customized}[1][]
{%  \usebackgroundtemplate{\includegraphics[width=\paperwidth,height=\paperheight]{kulakTitlepage.pdf}}
\hspace*{.38\textwidth}
\begin{minipage}{.62\textwidth}
\usebeamerfont{title}\usebeamercolor[fg]{title}{\inserttitle}~\\[6pt]
\usebeamerfont{subtitle}\usebeamercolor[fg]{subtitle}{\insertsubtitle}~\\[12pt]
\usebeamerfont{author}{\insertauthor}~\\[12pt]
\usebeamerfont{institute}{\insertinstitute}~\\[6pt]
\usebeamerfont{date}{\insertdate}~
\end{minipage}
}

\setbeamercolor{frametitle} {fg=kulblauw}%
\setbeamercolor{structure} {fg=kulblauw}%
\setbeamercolor{normal text} {fg=kulblauw}%
\setbeamercolor{alerted text} {fg=kuloranje}
\setbeamercolor{example text} {fg=kulgroen}

\setbeamercovered{transparent}
\setbeamercolor{author in head/foot}{fg=kulgroen!50!white}
\setbeamertemplate{section in head/foot}{\color{white}\bf\insertsectionhead}
\setbeamertemplate{section in head/foot shaded}{\color{kulgroen!50!white}\insertsectionhead}
\setbeamercolor{section in toc shaded}{fg=kulblauw}
\setbeamercolor{subsection in toc shaded}{fg=kulblauw}
\setbeamercolor{section in toc}{fg=white}
\setbeamercolor{subsection in toc}{fg=white}

\defbeamertemplate*{footline}{miniframes theme}
{%
  \begin{beamercolorbox}{section in head/foot}
    \vskip2pt
    %\insertnavigation{\paperwidth}
    \insertsectionnavigationhorizontal{\textwidth}{\hskip2ex}{\hskip2ex}
        \vskip2pt
  \end{beamercolorbox}%
}


\mode
<handout>
\logo{}
\newenvironment{titleframe}
  {\begin{frame}[plain]}
  {\centering\Logo\end{frame}}
\newenvironment{outlineframe}[1][]
  {\begin{frame}{#1}}
  {\end{frame}}
\usecolortheme{dove}


\mode
<all>
\beamertemplatenavigationsymbolsempty
\defbeamertemplate*{headline}{infolines theme}
{
  \leavevmode%
  \hbox{%
  \begin{beamercolorbox}[wd=\paperwidth,ht=2.25ex,dp=1ex,right,leftskip=2ex,rightskip=2ex]{author in head/foot}%
    \usebeamerfont{author in head/foot}\insertshortauthor \hfill
    %\insertshortinstitute \hfill
    \usebeamerfont{title in head/foot}\insertshorttitle \hfill
    %\usebeamerfont{date in head/foot}\insertshortdate{} 
    \usebeamerfont{date in head/foot}\insertframenumber{} / \inserttotalframenumber
    \end{beamercolorbox}}%
  \vskip0pt%
}